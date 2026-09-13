package de.tudarmstadt.physics.trackingplot.tracker

import de.tudarmstadt.physics.trackingplot.model.HsvColor
import de.tudarmstadt.physics.trackingplot.model.HsvTolerance
import org.opencv.core.Mat
import org.opencv.core.Rect

/** Result of one tracker for one frame, in pixel coordinates of the analysed frame. */
data class PixelTrackerResult(
    val trackerId: Int,
    val found: Boolean,
    val x: Int,
    val y: Int,
    val width: Int,
    val height: Int,
    val centroidX: Double,
    val centroidY: Double
)

/**
 * Kotlin facade over the native colour tracker. Every instance owns its own native state, so
 * different screens never interfere with each other. All methods are thread-safe; [close] must be
 * called exactly once when the tracker is no longer needed.
 */
class NativeTracker : AutoCloseable {

    private val lock = Any()
    private var handle: Long = nativeCreate()

    fun setTracker(id: Int, color: HsvColor, tolerance: HsvTolerance) {
        synchronized(lock) {
            if (handle == 0L) return
            nativeSetTracker(handle, id, color.h, color.s, color.v, tolerance.hue, tolerance.sat, tolerance.value)
        }
    }

    fun removeTracker(id: Int): Boolean = synchronized(lock) {
        if (handle == 0L) false else nativeRemoveTracker(handle, id)
    }

    fun clear() {
        synchronized(lock) {
            if (handle == 0L) return
            nativeClear(handle)
        }
    }

    /**
     * Runs all trackers on [frame] (CV_8UC4 RGBA or CV_8UC3 RGB). When [roi] is given only that
     * region is analysed; results are still in full-frame pixel coordinates.
     * Returns one entry per configured tracker, ordered by id.
     */
    fun update(frame: Mat, roi: Rect?): List<PixelTrackerResult> {
        val raw = synchronized(lock) {
            if (handle == 0L) return emptyList()
            nativeUpdate(
                handle, frame.nativeObjAddr,
                roi?.x ?: -1, roi?.y ?: -1, roi?.width ?: -1, roi?.height ?: -1
            )
        } ?: return emptyList()

        val count = raw.size / VALUES_PER_RESULT
        return List(count) { i ->
            val base = i * VALUES_PER_RESULT
            PixelTrackerResult(
                trackerId = raw[base].toInt(),
                found = raw[base + 1] != 0.0,
                x = raw[base + 2].toInt(),
                y = raw[base + 3].toInt(),
                width = raw[base + 4].toInt(),
                height = raw[base + 5].toInt(),
                centroidX = raw[base + 6],
                centroidY = raw[base + 7]
            )
        }
    }

    override fun close() {
        synchronized(lock) {
            val h = handle
            handle = 0L
            if (h != 0L) nativeDestroy(h)
        }
    }

    private companion object {
        const val VALUES_PER_RESULT = 8

        @JvmStatic
        external fun nativeCreate(): Long

        @JvmStatic
        external fun nativeDestroy(handle: Long)

        @JvmStatic
        external fun nativeSetTracker(
            handle: Long, id: Int,
            h: Int, s: Int, v: Int,
            hueTolerance: Int, satTolerance: Int, valTolerance: Int
        )

        @JvmStatic
        external fun nativeRemoveTracker(handle: Long, id: Int): Boolean

        @JvmStatic
        external fun nativeClear(handle: Long)

        @JvmStatic
        external fun nativeUpdate(
            handle: Long, matAddr: Long,
            roiX: Int, roiY: Int, roiWidth: Int, roiHeight: Int
        ): DoubleArray?
    }
}
