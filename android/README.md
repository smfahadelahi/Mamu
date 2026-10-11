# Mobile LLM — Android app

On-device chat with a GGUF model, running llama.cpp through JNI on arm64 Android phones
(Android 8.0 / API 26 and newer).

## Download the APK

1. Open the repository's **Releases** page and download `MobileLLM-debug.apk` from the latest release.
   Every build also appears under **Actions → Android APK → Artifacts**.
2. On the phone, allow installing from your browser or file manager, then open the APK.
   The build is debug-signed, so Android may ask you to confirm the install.
3. Open **Mobile LLM** and tap **Choose GGUF model file**. Pick a `.gguf` file from your device.
   The app copies it into its private storage (a multi-GB copy can take a minute) and loads it.

The APK does not include a model. Any llama.cpp-compatible GGUF with the `llama`
architecture works. Q4_K_M files are a good fit for a 6–8 GB phone.

## Build it yourself

Requirements: Android Studio (recent) or the command line with JDK 17, Android SDK
platform 35, build-tools 35.0.0, NDK 26.3.11579264, CMake 3.22.1.

```bash
# 1. Fetch llama.cpp at the pinned commit (CI does the same; the folder is git-ignored)
git init android/app/src/main/cpp/llama.cpp
git -C android/app/src/main/cpp/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git
git -C android/app/src/main/cpp/llama.cpp fetch --depth 1 origin 23b0202a189c44a54625aadcb37a946dd1d6278d
git -C android/app/src/main/cpp/llama.cpp checkout --detach FETCH_HEAD

# 2. Build the debug APK (Gradle 8.10.2; install it or run `gradle wrapper` once)
cd android && gradle :app:assembleDebug
# Output: android/app/build/outputs/apk/debug/app-debug.apk
```

Open the `android/` folder in Android Studio to run the app on a connected phone.

## Layout

```
android/
├── settings.gradle.kts, build.gradle.kts, gradle.properties
├── gradle/libs.versions.toml          # all dependency versions
└── app/
    ├── build.gradle.kts               # NDK, CMake, Compose, arm64-v8a only
    └── src/main/
        ├── AndroidManifest.xml
        ├── cpp/
        │   ├── CMakeLists.txt         # builds llama.cpp + libllm_engine.so
        │   ├── llm_bridge.cpp         # JNI: init, streamed generate, cancel, release
        │   └── llama.cpp/             # fetched at build time (git-ignored)
        ├── java/com/example/llm/
        │   ├── LlmEngine.kt           # Flow<String> wrapper over the native engine
        │   ├── LlmViewModel.kt        # model import/load, generation state
        │   └── MainActivity.kt        # Compose chat UI
        └── res/                       # icons, theme, strings
```

## Notes

- **Memory:** the model is memory-mapped, so Android can page weights in and out. A 2048-token
  context adds a KV cache of about 128 MB for the 1.5B architecture. Phones under about 6 GB of RAM
  may kill the app while a large model is loading.
- **Threads:** the engine uses up to 4 threads.
- **Conversation:** each prompt starts a fresh conversation. History is not carried over.
- **Process isolation:** inference runs in the app process. A native crash closes the app.
  Running inference in a separate service process is a planned change.
- **Performance:** ARMv8.2 dot-product and FP16 kernels are used. Older ARMv8.0 phones need
  `-DLLM_ARM_ARCH=armv8-a+dotprod` in `app/build.gradle.kts`, or the app will crash with SIGILL.
- **Signing:** the APK uses the default debug key. For Play Store distribution, add a release
  signing config and a keystore kept out of Git.
