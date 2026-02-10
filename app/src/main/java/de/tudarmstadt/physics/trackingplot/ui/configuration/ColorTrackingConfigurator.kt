package de.tudarmstadt.physics.trackingplot.ui.configuration

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

@Composable
fun ColorTrackingConfigurator(
    modifier: Modifier = Modifier,
    initialConfig: ColorTrackingConfig = ColorTrackingConfig(
        hue = 0f,
        saturation = 1f,
        value = 1f,
        hueThreshold = 10f,
        satThreshold = 0.3f,
        valThreshold = 0.3f
    ),
    onConfigChanged: (ColorTrackingConfig) -> Unit
) {
    var hue by remember { mutableStateOf(initialConfig.hue) }
    var saturation by remember { mutableStateOf(initialConfig.saturation) }
    var value by remember { mutableStateOf(initialConfig.value) }

    var hueThreshold by remember { mutableStateOf(initialConfig.hueThreshold) }
    var satThreshold by remember { mutableStateOf(initialConfig.satThreshold) }
    var valThreshold by remember { mutableStateOf(initialConfig.valThreshold) }

    LaunchedEffect(
        hue, saturation, value,
        hueThreshold, satThreshold, valThreshold
    ) {
        onConfigChanged(
            ColorTrackingConfig(
                hue = hue,
                saturation = saturation,
                value = value,
                hueThreshold = hueThreshold,
                satThreshold = satThreshold,
                valThreshold = valThreshold
            )
        )
    }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        Text("Tracked Color", style = MaterialTheme.typography.titleMedium)

        Spacer(Modifier.height(8.dp))

        ColorPreview(hue, saturation, value)

        Spacer(Modifier.height(16.dp))

        HueSlider(hue) { hue = it }
        SaturationSlider(saturation) { saturation = it }
        ValueSlider(value) { value = it }

        Spacer(Modifier.height(24.dp))

        Text("Thresholds", style = MaterialTheme.typography.titleMedium)

        Spacer(Modifier.height(8.dp))

        ThresholdSlider(
            label = "Hue ± ${hueThreshold.toInt()}°",
            value = hueThreshold,
            range = 1f..60f
        ) { hueThreshold = it }

        ThresholdSlider(
            label = "Saturation ± ${(satThreshold * 100).toInt()}%",
            value = satThreshold,
            range = 0.05f..1f
        ) { satThreshold = it }

        ThresholdSlider(
            label = "Value ± ${(valThreshold * 100).toInt()}%",
            value = valThreshold,
            range = 0.05f..1f
        ) { valThreshold = it }
    }
}

@Composable
private fun ColorPreview(h: Float, s: Float, v: Float) {
    val color = Color.hsv(h, s, v)

    Box(
        modifier = Modifier
            .size(72.dp)
            .clip(CircleShape)
            .background(color)
            .border(2.dp, Color.Black, CircleShape)
    )
}

@Composable
private fun HueSlider(
    value: Float,
    onValueChange: (Float) -> Unit
) {
    Column {
        Text("Hue (${value.toInt()}°)")
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = 0f..360f
        )
    }
}

@Composable
private fun SaturationSlider(
    value: Float,
    onValueChange: (Float) -> Unit
) {
    Column {
        Text("Saturation ${(value * 100).toInt()}%")
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = 0f..1f
        )
    }
}

@Composable
private fun ValueSlider(
    value: Float,
    onValueChange: (Float) -> Unit
) {
    Column {
        Text("Value ${(value * 100).toInt()}%")
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = 0f..1f
        )
    }
}

@Composable
private fun ThresholdSlider(
    label: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    onValueChange: (Float) -> Unit
) {
    Column {
        Text(label)
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = range
        )
    }
}

//fun ColorTrackingConfig.toOpenCvRanges(): Pair<Scalar, Scalar> {
//    val lower = Scalar(
//        (hue - hueThreshold).coerceIn(0f, 360f) / 2, // OpenCV uses 0..180
//        ((saturation - satThreshold).coerceIn(0f, 1f)) * 255,
//        ((value - valThreshold).coerceIn(0f, 1f)) * 255
//    )
//
//    val upper = Scalar(
//        (hue + hueThreshold).coerceIn(0f, 360f) / 2,
//        ((saturation + satThreshold).coerceIn(0f, 1f)) * 255,
//        ((value + valThreshold).coerceIn(0f, 1f)) * 255
//    )
//
//    return lower to upper
//}
