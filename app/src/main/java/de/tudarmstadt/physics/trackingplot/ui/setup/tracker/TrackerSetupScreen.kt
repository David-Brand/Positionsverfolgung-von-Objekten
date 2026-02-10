package de.tudarmstadt.physics.trackingplot.ui.setup.tracker

import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import de.tudarmstadt.physics.trackingplot.ui.ObserveAsEvents
import de.tudarmstadt.physics.trackingplot.ui.setup.SetupViewModel

@Composable
fun TrackerSetupScreen(
    toLiveExperiment: (experimentId: Long) -> Unit,
    setupViewModel: SetupViewModel
) {
    ObserveAsEvents(flow = setupViewModel.eventsChannelFlow) { event ->
        when (event) {
            is SetupViewModel.UiEvent.ToLiveExperiment -> {
                toLiveExperiment(event.experimentId)
            }
        }
    }

    Button(onClick = {
        setupViewModel.storeExperimentSetupAndStart()
    }) {
        Text("Start Experiment")
    }
}