package de.tudarmstadt.physics.trackingplot.ui.setup.tracker

data class NormalizedBox(
    val trackerId: Int,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val centroidX: Double,
    val centroidY: Double
)

data class NormalizedBox2(
    val left: Double,
    val top: Double,
    val right: Double,
    val bottom: Double
)

data class NormalizedCentroid(
    val centroidX: Double,
    val centroidY: Double
)