package de.tudarmstadt.physics.trackingplot.ui.configuration

data class ColorTrackingConfig(
    val hue: Float,          // 0..360
    val saturation: Float,   // 0..1
    val value: Float,        // 0..1
    val hueThreshold: Float, // +/- degrees
    val satThreshold: Float, // +/- 0..1
    val valThreshold: Float  // +/- 0..1
)
