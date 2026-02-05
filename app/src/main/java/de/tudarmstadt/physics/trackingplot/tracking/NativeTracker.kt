package de.tudarmstadt.physics.trackingplot.tracking

object NativeTracker {

    init {
        System.loadLibrary("native-lib")
    }

    external fun processFrame(
        rgbaMatAddr: Long,
        width: Int,
        height: Int
    ): FloatArray?
}