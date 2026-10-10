# Android app: on-device LLM (llama.cpp via JNI)

Layout:

```
android/app/
├── build.gradle.kts
└── src/main/
    ├── cpp/
    │   ├── CMakeLists.txt
    │   ├── llm_bridge.cpp        # JNI bridge (llama.cpp API, current header)
    │   └── llama.cpp/            # git submodule (not vendored here)
    └── java/com/example/llm/
        ├── LlmEngine.kt          # Flow<String> wrapper
        └── ChatViewModel.kt
```

## Setup

```bash
git submodule add https://github.com/ggml-org/llama.cpp android/app/src/main/cpp/llama.cpp
cd android/app/src/main/cpp/llama.cpp && git checkout <commit you have tested>
```

Requirements: Android NDK r26+ (see `build.gradle.kts`), CMake 3.22.1, `minSdk` 26,
and a `gradle/libs.versions.toml` that defines the `libs.*` aliases used in the build file.

The bridge was syntax-checked against llama.cpp `include/llama.h` (master, 2026-10-11).
It has not yet been built with the NDK or run on a device. Test the build on a real
arm64 phone before relying on it.

## Model file

Place a GGUF file (for example `model-q4_k_m.gguf`) in `context.filesDir`, then:

```kotlin
val engine = LlmEngine(File(context.filesDir, "model-q4_k_m.gguf").absolutePath)
```

Cancel active `generate` collections before calling `engine.close()`.
