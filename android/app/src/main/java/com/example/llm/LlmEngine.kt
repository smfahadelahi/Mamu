package com.example.llm

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import java.io.Closeable
import java.nio.charset.StandardCharsets
import java.util.concurrent.atomic.AtomicLong

/**
 * Thin Kotlin wrapper over the native llama.cpp bridge.
 *
 * Each [generate] call is a fresh conversation (the KV cache is cleared).
 * Call [close] when finished; the native engine is freed exactly once.
 */
class LlmEngine(
    modelPath: String,
    threads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4),
    contextSize: Int = 2048,
) : Closeable {

    private val handle = AtomicLong(0L)

    init {
        System.loadLibrary("llm_engine")
        val h = nativeInit(modelPath, threads, contextSize)
        require(h != 0L) { "Failed to load model from $modelPath" }
        handle.set(h)
    }

    /** Receives UTF-8 bytes containing only complete code points. */
    fun interface ByteCallback {
        fun onBytes(bytes: ByteArray)
    }

    /**
     * Streams generated text. Cancelling the collector stops native generation.
     * Emits nothing if the engine is already closed.
     */
    fun generate(prompt: String, maxTokens: Int = 512): Flow<String> = callbackFlow {
        val h = handle.get()
        if (h == 0L) {
            channel.close()
            awaitClose { }
            return@callbackFlow
        }

        val worker = launch(Dispatchers.Default) {
            try {
                nativeGenerate(h, prompt, maxTokens) { bytes ->
                    // Blocking send applies backpressure rather than dropping tokens.
                    trySendBlocking(String(bytes, StandardCharsets.UTF_8))
                }
            } finally {
                channel.close()
            }
        }

        awaitClose {
            nativeCancel(h)
            worker.cancel()
        }
    }.flowOn(Dispatchers.Default)

    /**
     * Frees native resources. Cancel any active [generate] collection first:
     * the native handle must not be used after this returns.
     */
    override fun close() {
        val h = handle.getAndSet(0L)
        if (h != 0L) nativeRelease(h)
    }

    private external fun nativeInit(modelPath: String, threads: Int, contextSize: Int): Long
    private external fun nativeGenerate(handle: Long, prompt: String, maxTokens: Int, callback: ByteCallback)
    private external fun nativeCancel(handle: Long)
    private external fun nativeRelease(handle: Long)
}
