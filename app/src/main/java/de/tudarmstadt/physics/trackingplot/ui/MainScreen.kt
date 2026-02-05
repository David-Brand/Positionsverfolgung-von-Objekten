package de.tudarmstadt.physics.trackingplot.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Checkbox
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import de.tudarmstadt.physics.trackingplot.ui.plotting.PositionSample
import de.tudarmstadt.physics.trackingplot.ui.plotting.TimePoint
import de.tudarmstadt.physics.trackingplot.ui.plotting.TimeSeriesChartMulti
import de.tudarmstadt.physics.trackingplot.ui.tracking.ColoredObjectCameraView
import kotlin.math.exp
import kotlin.math.ln

@Composable
fun MainScreen(
    viewModel: CameraChartViewModel = viewModel()
) {

////    val samples = remember { mutableStateListOf<PositionSample>() }
//
//    val xSeries = remember { mutableStateListOf<TimePoint>() }
//    val ySeries = remember { mutableStateListOf<TimePoint>() }
//
//
//    // Append helper
////    fun addSample(sample: PositionSample) {
////        samples.add(sample)
////        xSeries.add(TimePoint(sample.timeMs, sample.x))
////        ySeries.add(TimePoint(sample.timeMs, sample.y))
////
////        // keep last 10 minutes
////        val cutoff = sample.timeMs - 10 * 60_000
////        while (xSeries.firstOrNull()?.timeMs ?: Long.MAX_VALUE < cutoff) {
////            xSeries.removeAt(0)
////            ySeries.removeAt(0)
////        }
////    }
//    fun onPositionDetected(sample: PositionSample) {
//        xSeries.add(TimePoint(sample.timeMs, sample.x))
//        ySeries.add(TimePoint(sample.timeMs, sample.y))
//
//        // prune old samples if needed
//        val cutoff = sample.timeMs - 10 * 60_000
//        while (xSeries.firstOrNull()?.timeMs ?: Long.MAX_VALUE < cutoff) {
//            xSeries.removeAt(0)
//            ySeries.removeAt(0)
//        }
//    }



    var plotX by remember { mutableStateOf(true) }
    var plotY by remember { mutableStateOf(false) }

    Column(Modifier.fillMaxSize()) {

        // ───── TOP: CAMERA ─────
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
        ) {
            ColoredObjectCameraView(
                onPositionDetected = { sample ->
                    viewModel.addSample(sample)
                }
            )
        }

        // ───── BOTTOM: CHART ─────
        Column(
            modifier = Modifier
                .weight(1f)
                .padding(8.dp)
        ) {

            Row {
                Checkbox(checked = plotX, onCheckedChange = { plotX = it })
                Text("Plot X")

                Spacer(Modifier.width(16.dp))

                Checkbox(checked = plotY, onCheckedChange = { plotY = it })
                Text("Plot Y")
            }

            var windowSlider by remember { mutableFloatStateOf(0.4f) }
            var offsetSlider by remember { mutableFloatStateOf(0f) }

            val windowMs = remember(windowSlider) {
                windowFromSlider(windowSlider)
            }

// Compute total duration from the live series safely
            val totalDurationMs = remember(viewModel.xSeries, viewModel.ySeries) {
                val allPoints = (viewModel.xSeries + viewModel.ySeries)
                if (allPoints.isEmpty()) 0L
                else allPoints.maxOf { it.timeMs } - allPoints.minOf { it.timeMs }
            }

// Compute maxOffset dynamically
            val maxOffsetMs = (totalDurationMs - windowMs).coerceAtLeast(0L)

// Compute actual offset in ms
            val offsetMs = remember(offsetSlider, windowMs, totalDurationMs) {
                (offsetSlider * maxOffsetMs).toLong()
            }

//            var windowSlider by remember { mutableFloatStateOf(0.4f) }
//            var offsetSlider by remember { mutableFloatStateOf(0f) }
//
//            val windowMs = remember(windowSlider) {
//                windowFromSlider(windowSlider)
//            }
//
//            val totalDurationMs = remember(viewModel.xSeries, viewModel.ySeries) {
//                val allPoints = (viewModel.xSeries + viewModel.ySeries)
//                    .takeIf { it.isNotEmpty() }
//                    ?.let { it.last().timeMs - it.first().timeMs }
//                    ?: 0L
//                allPoints
//            }
//
//            val maxOffsetMs =
//                (totalDurationMs - windowMs).coerceAtLeast(0)
//
//            val offsetMs = remember(offsetSlider, windowMs, totalDurationMs) {
//                (offsetSlider * maxOffsetMs).toLong()
//            }

            TimeSeriesChartMulti(
//                xPoints = if (plotX) xSeries else emptyList(),
//                yPoints = if (plotY) ySeries else emptyList(),
                xPoints = if (plotX) viewModel.xSeries else emptyList(),
                yPoints = if (plotY) viewModel.ySeries else emptyList(),
                visibleWindowMs = windowMs,
                windowOffsetMs = offsetMs,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(260.dp)
            )

            // your existing sliders go here
            Spacer(Modifier.height(16.dp))

            Text("Time window: ${windowMs / 1000}s")
            Slider(
                value = windowSlider,
                onValueChange = { windowSlider = it }
            )

            Spacer(Modifier.height(8.dp))

            Text(
                if (offsetMs == 0L) "Live"
                else "Offset: ${offsetMs / 1000}s"
            )
            Slider(
                value = offsetSlider,
                onValueChange = { offsetSlider = it }
            )
        }
    }
}

fun windowFromSlider(value: Float): Long {
    val minMs = 2_000L       // 2 seconds
    val maxMs = 30 * 60_000L // 30 minutes

    val logMin = ln(minMs.toDouble())
    val logMax = ln(maxMs.toDouble())

    return exp(logMin + value * (logMax - logMin)).toLong()
}
