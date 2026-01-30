package de.tudarmstadt.physics.trackingplot.ui.plotting

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay
import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random

@Composable
fun TimeSeriesDemo() {
    val dataBuffer = remember { mutableStateListOf<TimePoint>() }
    val scope = rememberCoroutineScope()

    // Dummy data generator
    LaunchedEffect(Unit) {
        var t = 0f
        while (true) {
            val now = System.currentTimeMillis()
            val value =
                kotlin.math.sin(t) * 10f + Random.nextFloat() * 2f

            dataBuffer.add(TimePoint(now, value))
            t += 0.15f

            // Keep last 24h
            val cutoff = now - 24 * 60 * 60 * 1000L
            while (dataBuffer.firstOrNull()?.timeMs ?: Long.MAX_VALUE < cutoff) {
                dataBuffer.removeAt(0)
            }

            delay(50) // 20 Hz
        }
    }

    var windowSlider by remember { mutableFloatStateOf(0.4f) }
    var offsetSlider by remember { mutableFloatStateOf(0f) }

    val windowMs = remember(windowSlider) {
        windowFromSlider(windowSlider)
    }


    val totalDurationMs =
        if (dataBuffer.size < 2) 0L
        else dataBuffer.last().timeMs - dataBuffer.first().timeMs

    val maxOffsetMs =
        (totalDurationMs - windowMs).coerceAtLeast(0)

    val offsetMs = remember(offsetSlider, windowMs, totalDurationMs) {
        (offsetSlider * maxOffsetMs).toLong()
    }
//    val maxOffsetMs = remember(dataBuffer.size) {
//        if (dataBuffer.isEmpty()) 0L
//        else System.currentTimeMillis() - dataBuffer.first().timeMs
//    }

//    val offsetMs = remember(offsetSlider, maxOffsetMs) {
//        (offsetSlider * maxOffsetMs).toLong()
//    }

    Column(modifier = Modifier.padding(16.dp)) {
        TimeSeriesChart(
            points = dataBuffer,
            visibleWindowMs = windowMs,
            windowOffsetMs = offsetMs,
            modifier = Modifier
                .fillMaxWidth()
                .height(320.dp)
        )

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

//fun windowFromSlider(value: Float): Long {
//    val minMs = 5_000L                 // 5 seconds
//    val maxMs = 24 * 60 * 60 * 1000L   // 24 hours
//
//    val logMin = ln(minMs.toDouble())
//    val logMax = ln(maxMs.toDouble())
//
//    val logValue = logMin + value * (logMax - logMin)
//    return exp(logValue).toLong()
//}

fun windowFromSlider(value: Float): Long {
    val minMs = 2_000L       // 2 seconds
    val maxMs = 30 * 60_000L // 30 minutes

    val logMin = ln(minMs.toDouble())
    val logMax = ln(maxMs.toDouble())

    return exp(logMin + value * (logMax - logMin)).toLong()
}
