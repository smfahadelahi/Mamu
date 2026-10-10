// JNI bridge between Kotlin (LlmEngine) and llama.cpp.
//
// Written against llama.cpp's include/llama.h as of 2026-10 (post-"llama_model_load_from_file" API).
// Pin the llama.cpp submodule to a commit you have built and tested.

#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"

#define TAG "LLMBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

struct EngineContext {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    llama_sampler *sampler = nullptr;
    uint32_t n_ctx = 0;
    std::mutex run_mutex;             // serialises generate() and release()
    std::atomic<bool> cancel{false};  // set from Kotlin to stop generation early
};

void ensure_backend_initialized() {
    // llama_backend_init must run once per process, not per engine instance.
    static std::once_flag once;
    std::call_once(once, [] { llama_backend_init(); });
}

// Length of a trailing, incomplete UTF-8 sequence in buf (0 if none).
// Used so that multi-byte characters split across tokens are not emitted half-formed.
size_t incomplete_utf8_tail(const std::string &buf) {
    const size_t n = buf.size();
    for (size_t back = 1; back <= 4 && back <= n; ++back) {
        const auto c = static_cast<unsigned char>(buf[n - back]);
        if ((c & 0xC0) == 0x80) continue;  // continuation byte, keep scanning back
        size_t need = 1;
        if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        return need > back ? back : 0;
    }
    return 0;
}

bool tokenize_prompt(const llama_vocab *vocab, const std::string &text, std::vector<llama_token> &out) {
    // Passing a null buffer returns the negative of the required token count.
    const int32_t needed = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                          nullptr, 0, /*add_special=*/true, /*parse_special=*/true);
    const int32_t count = needed < 0 ? -needed : needed;
    out.resize(count);
    const int32_t written = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                           out.data(), count, true, true);
    if (written < 0) return false;
    out.resize(written);
    return true;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_example_llm_LlmEngine_nativeInit(JNIEnv *env, jobject /* thiz */,
                                          jstring model_path_jstr, jint n_threads, jint n_ctx) {
    ensure_backend_initialized();

    const char *model_path = env->GetStringUTFChars(model_path_jstr, nullptr);
    if (!model_path) return 0;
    LOGI("Loading model from: %s", model_path);

    llama_model_params mparams = llama_model_default_params();
    // Memory-map the GGUF file so the kernel can page weights in and out.
    mparams.load_mode = LLAMA_LOAD_MODE_MMAP;
    llama_model *model = llama_model_load_from_file(model_path, mparams);
    env->ReleaseStringUTFChars(model_path_jstr, model_path);

    if (!model) {
        LOGE("Failed to load model file");
        return 0;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(n_ctx);
    cparams.n_batch = static_cast<uint32_t>(n_ctx);
    cparams.n_ubatch = 256;  // smaller physical batches keep peak scratch memory low
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOGE("Failed to create context");
        llama_model_free(model);
        return 0;
    }

    // Sampler chain: top-p -> temperature -> stochastic draw.
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    auto *engine = new EngineContext();
    engine->model = model;
    engine->ctx = ctx;
    engine->sampler = sampler;
    engine->n_ctx = llama_n_ctx(ctx);
    return reinterpret_cast<jlong>(engine);
}

// Streams UTF-8 bytes to `callback.onBytes(ByteArray)`. Each call contains only
// complete code points, so Kotlin can decode it with String(bytes, UTF_8).
JNIEXPORT void JNICALL
Java_com_example_llm_LlmEngine_nativeGenerate(JNIEnv *env, jobject /* thiz */, jlong handle,
                                              jstring prompt_jstr, jint max_tokens, jobject callback) {
    auto *engine = reinterpret_cast<EngineContext *>(handle);
    if (!engine || !prompt_jstr || !callback) return;

    std::lock_guard<std::mutex> lock(engine->run_mutex);
    engine->cancel.store(false);

    jclass cb_class = env->GetObjectClass(callback);
    jmethodID on_bytes = env->GetMethodID(cb_class, "onBytes", "([B)V");
    env->DeleteLocalRef(cb_class);
    if (!on_bytes) return;

    const char *prompt_chars = env->GetStringUTFChars(prompt_jstr, nullptr);
    if (!prompt_chars) return;
    std::string prompt(prompt_chars);
    env->ReleaseStringUTFChars(prompt_jstr, prompt_chars);

    const llama_vocab *vocab = llama_model_get_vocab(engine->model);

    std::vector<llama_token> prompt_tokens;
    if (!tokenize_prompt(vocab, prompt, prompt_tokens)) {
        LOGE("Tokenization failed");
        return;
    }
    if (prompt_tokens.size() + 1 >= engine->n_ctx) {
        LOGE("Prompt of %zu tokens does not fit in context of %u", prompt_tokens.size(), engine->n_ctx);
        return;
    }

    // Each call is a fresh conversation. Clearing the KV cache prevents
    // silent accumulation across calls, which would overflow n_ctx.
    llama_memory_clear(llama_get_memory(engine->ctx), true);
    llama_sampler_reset(engine->sampler);

    // Decode the prompt in chunks no larger than n_batch.
    const size_t chunk = llama_n_batch(engine->ctx);
    for (size_t off = 0; off < prompt_tokens.size(); off += chunk) {
        const size_t n = std::min(chunk, prompt_tokens.size() - off);
        llama_batch batch = llama_batch_get_one(prompt_tokens.data() + off, static_cast<int32_t>(n));
        if (llama_decode(engine->ctx, batch) != 0) {
            LOGE("Failed to decode prompt chunk");
            return;
        }
    }

    uint32_t n_past = static_cast<uint32_t>(prompt_tokens.size());
    std::string pending;  // bytes not yet emitted (partial UTF-8 at the end)
    char piece[256];

    auto emit = [&](const std::string &bytes) -> bool {
        jbyteArray arr = env->NewByteArray(static_cast<jsize>(bytes.size()));
        if (!arr) return false;
        env->SetByteArrayRegion(arr, 0, static_cast<jsize>(bytes.size()),
                                reinterpret_cast<const jbyte *>(bytes.data()));
        env->CallVoidMethod(callback, on_bytes, arr);
        env->DeleteLocalRef(arr);
        return !env->ExceptionCheck();  // stop if Kotlin threw
    };

    for (int32_t i = 0; i < max_tokens; ++i) {
        if (engine->cancel.load() || n_past + 1 >= engine->n_ctx) break;

        llama_token id = llama_sampler_sample(engine->sampler, engine->ctx, -1);
        llama_sampler_accept(engine->sampler, id);
        if (llama_vocab_is_eog(vocab, id)) break;

        int32_t n = llama_token_to_piece(vocab, id, piece, sizeof(piece), 0, /*special=*/false);
        if (n < 0) {
            // Piece longer than the stack buffer: retry with the exact size.
            std::string big(static_cast<size_t>(-n), '\0');
            n = llama_token_to_piece(vocab, id, big.data(), static_cast<int32_t>(big.size()), 0, false);
            if (n > 0) pending.append(big.data(), static_cast<size_t>(n));
        } else if (n > 0) {
            pending.append(piece, static_cast<size_t>(n));
        }

        const size_t tail = incomplete_utf8_tail(pending);
        const size_t ready = pending.size() - tail;
        if (ready > 0) {
            if (!emit(pending.substr(0, ready))) break;
            pending.erase(0, ready);
        }

        llama_batch batch = llama_batch_get_one(&id, 1);
        if (llama_decode(engine->ctx, batch) != 0) {
            LOGE("Failed to decode generated token");
            break;
        }
        ++n_past;
    }

    // Flush whatever is left (only reached if generation ended mid-character).
    if (!pending.empty()) emit(pending);
}

JNIEXPORT void JNICALL
Java_com_example_llm_LlmEngine_nativeCancel(JNIEnv * /* env */, jobject /* thiz */, jlong handle) {
    auto *engine = reinterpret_cast<EngineContext *>(handle);
    if (engine) engine->cancel.store(true);
}

JNIEXPORT void JNICALL
Java_com_example_llm_LlmEngine_nativeRelease(JNIEnv * /* env */, jobject /* thiz */, jlong handle) {
    auto *engine = reinterpret_cast<EngineContext *>(handle);
    if (!engine) return;

    // Stop any in-flight generation, then wait for it to leave the lock.
    engine->cancel.store(true);
    std::lock_guard<std::mutex> lock(engine->run_mutex);

    if (engine->sampler) llama_sampler_free(engine->sampler);
    if (engine->ctx) llama_free(engine->ctx);
    if (engine->model) llama_model_free(engine->model);
    delete engine;
    LOGI("Native engine released");
}

}  // extern "C"
