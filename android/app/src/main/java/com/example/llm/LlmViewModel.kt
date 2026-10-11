package com.example.llm

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

enum class ModelState { NONE, IMPORTING, LOADING, READY, ERROR }

data class UiState(
    val modelState: ModelState = ModelState.NONE,
    val status: String = "No model loaded. Choose a .gguf file to begin.",
    val output: String = "",
    val generating: Boolean = false,
)

class LlmViewModel(app: Application) : AndroidViewModel(app) {

    private val modelFile = File(app.filesDir, MODEL_FILE_NAME)

    // Outlives the ViewModel so native cleanup can finish after onCleared().
    private val releaseScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val _ui = MutableStateFlow(UiState())
    val ui: StateFlow<UiState> = _ui.asStateFlow()

    // Only touched on Dispatchers.IO or the main thread via viewModelScope.
    @Volatile
    private var engine: LlmEngine? = null
    private var generation: Job? = null

    init {
        if (modelFile.exists()) loadModel()
    }

    /** Copies a user-picked .gguf file into app storage, then loads it. */
    fun importModel(uri: Uri) {
        _ui.update { it.copy(modelState = ModelState.IMPORTING, status = "Copying model...") }
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    releaseEngine()
                    val tmp = File(modelFile.parentFile, "$MODEL_FILE_NAME.part")
                    val input = getApplication<Application>().contentResolver.openInputStream(uri)
                        ?: error("Could not open the selected file")
                    input.use { src -> tmp.outputStream().use { dst -> src.copyTo(dst, 1 shl 20) } }
                    if (!tmp.renameTo(modelFile)) error("Could not save the model file")
                }
                loadModelInternal()
            } catch (e: Exception) {
                fail("Import failed: ${e.message ?: e.javaClass.simpleName}")
            }
        }
    }

    /** Sends a prompt. A new prompt cancels any generation still running. */
    fun ask(prompt: String) {
        val current = engine
        if (current == null) {
            fail("Load a model before asking a question.")
            return
        }
        generation?.cancel()
        _ui.update { it.copy(output = "", generating = true) }
        generation = viewModelScope.launch {
            try {
                current.generate(prompt).collect { piece ->
                    _ui.update { it.copy(output = it.output + piece) }
                }
            } catch (e: Exception) {
                // Keep the model loaded; only report the error.
                _ui.update { it.copy(status = "Generation failed: ${e.message ?: e.javaClass.simpleName}") }
            } finally {
                _ui.update { it.copy(generating = false) }
            }
        }
    }

    fun stop() {
        generation?.cancel()
    }

    private fun loadModel() {
        _ui.update { it.copy(modelState = ModelState.LOADING, status = "Loading model...") }
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) { loadModelInternal() }
            } catch (e: Exception) {
                fail("Load failed: ${e.message ?: e.javaClass.simpleName}")
            }
        }
    }

    /** Runs on Dispatchers.IO. Loads the native engine and reports READY. */
    private suspend fun loadModelInternal() {
        _ui.update { it.copy(modelState = ModelState.LOADING, status = "Loading model...") }
        releaseEngine()
        // Native init is blocking and memory-mapped; it can take several seconds.
        val loaded = withContext(Dispatchers.IO) {
            LlmEngine(modelFile.absolutePath, contextSize = CONTEXT_TOKENS)
        }
        engine = loaded
        _ui.update {
            it.copy(
                modelState = ModelState.READY,
                status = "Model ready: ${modelFile.name} (${modelFile.length() / (1024 * 1024)} MB)",
            )
        }
    }

    /** Stops any running generation, then frees the native engine. */
    private suspend fun releaseEngine() {
        generation?.cancelAndJoin()
        generation = null
        engine?.close()
        engine = null
    }

    private fun fail(message: String) {
        _ui.update { it.copy(modelState = ModelState.ERROR, status = message, generating = false) }
    }

    override fun onCleared() {
        // cancelAndJoin waits for native generation to exit before the engine is freed.
        releaseScope.launch { releaseEngine() }
        super.onCleared()
    }

    companion object {
        const val MODEL_FILE_NAME = "model.gguf"
        const val CONTEXT_TOKENS = 2048
    }
}
