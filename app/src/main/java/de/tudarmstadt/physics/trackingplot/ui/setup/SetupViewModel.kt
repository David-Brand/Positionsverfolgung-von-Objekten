package de.tudarmstadt.physics.trackingplot.ui.setup

import androidx.compose.runtime.mutableStateListOf
import androidx.compose.ui.geometry.Offset
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch

class SetupViewModel: ViewModel() {

    private val _uiPoints = mutableStateListOf<Offset>() //ui offset coordinates
    val uiPoints = _uiPoints as List<Offset>

    private val _normalizedPoints = mutableStateListOf<Pair<Float, Float>>() //0.0 - 1.0
    val normalizedPoints = _normalizedPoints as List<Pair<Float, Float>>

    fun storeExperimentSetupAndStart() {
        viewModelScope.launch {
            //todo store setup

            //on success
            val experimentId = 123L //todo this is returned by setup store

            eventsChannel.send(UiEvent.ToLiveExperiment(experimentId))
        }
    }


    private val eventsChannel = Channel<UiEvent>()
    val eventsChannelFlow = eventsChannel.receiveAsFlow()

    sealed interface UiEvent {
        data class ToLiveExperiment(val experimentId: Long): UiEvent
    }
}