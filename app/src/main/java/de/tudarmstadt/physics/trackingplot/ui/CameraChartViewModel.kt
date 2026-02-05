package de.tudarmstadt.physics.trackingplot.ui

import androidx.compose.runtime.mutableStateListOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import de.tudarmstadt.physics.trackingplot.ui.plotting.PositionSample
import de.tudarmstadt.physics.trackingplot.ui.plotting.TimePoint
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class CameraChartViewModel : ViewModel() {

    // SnapshotStateLists observed by Compose
    val xSeries = mutableStateListOf<TimePoint>()
    val ySeries = mutableStateListOf<TimePoint>()

    // Private thread-safe buffers
    private val xBuffer = mutableListOf<TimePoint>()
    private val yBuffer = mutableListOf<TimePoint>()

    // Call this from camera callback
    fun addSample(sample: PositionSample) {
        synchronized(xBuffer) {
            xBuffer.add(TimePoint(sample.timeMs, sample.x))
            yBuffer.add(TimePoint(sample.timeMs, sample.y))
        }
    }

    init {
        // Launch a coroutine to merge buffer -> SnapshotStateList
        viewModelScope.launch {
            while (true) {
                withContext(Dispatchers.Main) {
                    synchronized(xBuffer) {
                        if (xBuffer.isNotEmpty()) {
                            xSeries.addAll(xBuffer)
                            ySeries.addAll(yBuffer)
                            xBuffer.clear()
                            yBuffer.clear()
                        }
                    }
                    // Optional pruning to keep memory small
                    val cutoff = System.currentTimeMillis() - 10 * 60_000L
                    while (xSeries.firstOrNull()?.timeMs ?: Long.MAX_VALUE < cutoff) {
                        xSeries.removeAt(0)
                        ySeries.removeAt(0)
                    }
                }
                delay(16) // ~60 FPS merge
            }
        }
    }
}