package de.tudarmstadt.physics.trackingplot.model

/** One tracked object in one frame; all coordinates are canonical and normalised to 0..1. */
data class TrackedObject(
    val trackerId: Int,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val centroidX: Double,
    val centroidY: Double
)

/** The tracking result of one analysed camera frame. */
data class FrameResult(
    /** Sensor timestamp of the frame in nanoseconds (monotonic, see ImageInfo.getTimestamp). */
    val timestampNanos: Long,
    /** Only trackers that were found in this frame. */
    val objects: List<TrackedObject>
)

/** A stored measurement. */
data class MeasurementSample(
    val trackerIndex: Int,
    /** Milliseconds since the experiment was started. */
    val timestampMs: Long,
    val centroidX: Double,
    val centroidY: Double
)
