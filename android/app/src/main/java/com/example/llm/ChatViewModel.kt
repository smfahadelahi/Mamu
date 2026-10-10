package com.example.llm

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch

class ChatViewModel(private val engine: LlmEngine) : ViewModel() {

    private val _uiText = MutableStateFlow("")
    val uiText: StateFlow<String> = _uiText.asStateFlow()

    private var job: Job? = null

    fun ask(prompt: String) {
        job?.cancel()  // a new prompt replaces any in-flight generation
        _uiText.value = ""
        job = viewModelScope.launch {
            engine.generate(prompt).collect { piece ->
                _uiText.update { it + piece }
            }
        }
    }

    override fun onCleared() {
        // Cancel and wait for the native worker to exit before freeing the engine.
        // viewModelScope is already cancelled here, so use a separate scope.
        val running = job
        CoroutineScope(Dispatchers.IO).launch {
            running?.cancelAndJoin()
            engine.close()
        }
        super.onCleared()
    }
}
