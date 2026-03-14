package de.tudarmstadt.physics.trackingplot.tracker

data class TrackerResult(
    val trackerId: Int,
    val found: Boolean,
    val x: Int,
    val y: Int,
    val width: Int,
    val height: Int,
    val centroidX: Double,
    val centroidY: Double
)