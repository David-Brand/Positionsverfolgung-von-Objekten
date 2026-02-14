package de.tudarmstadt.physics.trackingplot.tracker

import org.opencv.core.Mat

class NativeTracker {

    companion object {
        init {
            System.loadLibrary("native-lib")
        }
    }

    external fun nativeInit(
        matAddr: Long,
        configs: Array<TrackerConfig>,
        lowerHSV: DoubleArray,
        upperHSV: DoubleArray
    )

    external fun nativeUpdate(
        matAddr: Long
    ): Array<TrackerResult>

    fun init(
        frame: Mat,
        configs: List<TrackerConfig>,
        lowerHSV: DoubleArray,
        upperHSV: DoubleArray
    ) {
        nativeInit(
            frame.nativeObjAddr,
            configs.toTypedArray(),
            lowerHSV,
            upperHSV
        )
    }

    fun update(frame: Mat): List<TrackerResult> {
        return nativeUpdate(frame.nativeObjAddr).toList()
    }
}