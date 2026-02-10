package de.tudarmstadt.physics.trackingplot.ui.setup

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch

class SetupViewModel: ViewModel() {

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