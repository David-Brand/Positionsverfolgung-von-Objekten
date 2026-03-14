package de.tudarmstadt.physics.trackingplot.tracker2

import de.tudarmstadt.physics.trackingplot.DistanceUnit

object TrackingSession {

    private var config: TrackingConfig? = null

    fun configure(newConfig: TrackingConfig) {
        config = newConfig
    }

    fun getConfig(): TrackingConfig =
        config ?: error("TrackingSession not configured")

    // ---- Active state ----
    @Volatile
    private var running = false

    fun start() { running = true }
    fun stop() { running = false }
    fun isActive() = running

    fun reset() {
        running = false
        config = null
    }
}

data class Point2D(val x: Float, val y: Float)

data class Roi(
    val p1: Point2D,
    val p2: Point2D
)

data class Ruler(
    val p1: Point2D,
    val p2: Point2D,
    val realDistance: Float,
    val unit: DistanceUnit
)

data class ColorTrackerConfig(
    val color: Int,          // ARGB
    val tolerance: Float     // e.g. Euclidean distance in color space
)

data class TrackingConfig(
    val roi: Roi?,
    val ruler: Ruler?,
    val trackers: List<ColorTrackerConfig> // size 1..3
)