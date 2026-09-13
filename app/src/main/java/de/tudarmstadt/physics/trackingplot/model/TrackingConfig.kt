package de.tudarmstadt.physics.trackingplot.model

/**
 * A point in the canonical camera frame, normalised to 0..1 in both axes.
 *
 * The canonical frame is the camera image rotated to portrait orientation with a fixed 3:4
 * (width:height) aspect ratio, independent of how the device is currently held.
 * x grows to the right and y grows downwards (image convention).
 */
data class Point2D(val x: Float, val y: Float)

/** Region of interest given by two opposite corners (any order) in canonical coordinates. */
data class Roi(val p1: Point2D, val p2: Point2D) {
    val left: Float get() = minOf(p1.x, p2.x)
    val top: Float get() = minOf(p1.y, p2.y)
    val right: Float get() = maxOf(p1.x, p2.x)
    val bottom: Float get() = maxOf(p1.y, p2.y)
}

/** Two reference points a known real-world distance apart, in canonical coordinates. */
data class Ruler(
    val p1: Point2D,
    val p2: Point2D,
    val realDistance: Float,
    val unit: DistanceUnit
)

data class ColorTrackerConfig(
    /** Picked colour as ARGB. */
    val color: Int,
    /** Tolerance as shown in the UI, 0..100 (see [HsvTolerance.fromUiTolerance]). */
    val tolerance: Int
) {
    companion object {
        const val DEFAULT_TOLERANCE = 40
        const val MIN_TOLERANCE = 5
        const val MAX_TOLERANCE = 100
        const val MAX_TRACKERS = 3
    }
}

data class TrackingConfig(
    val roi: Roi?,
    val ruler: Ruler?,
    /** 1..[ColorTrackerConfig.MAX_TRACKERS] trackers; the list index is the tracker id. */
    val trackers: List<ColorTrackerConfig>,
    /** Maximum number of stored samples per second and tracker; 0 = every analysed frame. */
    val samplingRate: Int,
    /** Camera zoom ratio used during setup; must be restored for the calibration to hold. */
    val zoomRatio: Float
) {
    companion object {
        const val DEFAULT_SAMPLING_RATE = 10
        const val MIN_SAMPLING_RATE = 1
        const val MAX_SAMPLING_RATE = 60
    }
}
