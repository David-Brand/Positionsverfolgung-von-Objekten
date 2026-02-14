package de.tudarmstadt.physics.trackingplot.ui.live_experiment

import androidx.lifecycle.ViewModel
import de.tudarmstadt.physics.trackingplot.tracker.NativeTracker
import de.tudarmstadt.physics.trackingplot.tracker.TrackerConfig

class LiveExperimentViewModel(
    val experimentId: Long
): ViewModel() {

    private val tracker: NativeTracker

    init {
        tracker = NativeTracker()
//        TrackerConfig(
//
//        )
    }
}