package de.tudarmstadt.physics.trackingplot.tracker

data class TrackerResult(
    val id: Int,
    val x: Double,
    val y: Double,
    val width: Double,
    val height: Double,
    val valid: Boolean
)