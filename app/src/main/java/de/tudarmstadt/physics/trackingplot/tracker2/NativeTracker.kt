package de.tudarmstadt.physics.trackingplot.tracker2

import android.graphics.Rect
import de.tudarmstadt.physics.trackingplot.tracker.TrackerResult

class NativeTracker {

    external fun addTracker(
//        h: Int,
//        s: Int,
//        v: Int,
        r: Int,
        g: Int,
        b: Int,
        tolerance: Int
    ): Int

    external fun setTracker(
//        h: Int,
//        s: Int,
//        v: Int,
        index: Int,
        r: Int,
        g: Int,
        b: Int,
        tolerance: Int
    ): Int

    external fun updateTrackers(
        matAddr: Long,
        roiX: Int,
        roiY: Int,
        roiWidth: Int,
        roiHeight: Int
    ): Array<TrackerResult>

    external fun reset()

    fun updateTrackers(
        matAddr: Long,
        roi: Rect? = null
    ): Array<TrackerResult> {
        return if (roi != null) {
            updateTrackers(
                matAddr,
                roi.left,
                roi.top,
                roi.width(),
                roi.height()
            )
        } else {
            updateTrackers(
                matAddr,
                -1, -1, -1, -1
            )
        }
    }
}