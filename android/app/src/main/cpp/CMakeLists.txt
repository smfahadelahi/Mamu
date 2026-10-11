cmake_minimum_required(VERSION 3.22.1)
project("mobile_llm_jni" C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp is missing. Clone it into app/src/main/cpp/llama.cpp "
        "(see android/README.md). CI does this automatically.")
endif()

# ---------------------------------------------------------------------------
# llama.cpp: only the library. Tools, tests, OpenSSL and the common helper library
# are not needed by the JNI bridge and would add build time and dependencies.
# ---------------------------------------------------------------------------
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP          OFF CACHE BOOL "" FORCE)  # Android's thread model works better without OpenMP

if (ANDROID_ABI STREQUAL "arm64-v8a")
    # ISA baseline for ggml's CPU kernels. armv8.2-a+dotprod+fp16 is missing on ARMv8.0
    # cores (e.g. Cortex-A53) and would crash with SIGILL there. Override with
    # -DLLM_ARM_ARCH=armv8-a+dotprod for broader compatibility.
    set(LLM_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "ARM ISA for ggml CPU kernels")
    set(GGML_CPU_ARM_ARCH "${LLM_ARM_ARCH}" CACHE STRING "" FORCE)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp)

# ---------------------------------------------------------------------------
# JNI bridge -> libllm_engine.so
# ---------------------------------------------------------------------------
add_library(llm_engine SHARED llm_bridge.cpp)

target_include_directories(llm_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
)

find_library(log-lib log)

target_link_libraries(llm_engine PRIVATE
    llama
    ${log-lib}
)
