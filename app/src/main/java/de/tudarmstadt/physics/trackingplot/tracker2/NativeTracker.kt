package de.tudarmstadt.physics.trackingplot.tracker2

import android.graphics.Rect

class NativeTracker {

    companion object {
        init {
            System.loadLibrary("native-lib")
        }
    }

    external fun addTracker(
//        h: Int,
//        s: Int,
//        v: Int,
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