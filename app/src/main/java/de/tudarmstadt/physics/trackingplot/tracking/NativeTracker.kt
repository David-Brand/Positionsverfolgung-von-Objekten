package de.tudarmstadt.physics.trackingplot.tracking

import java.nio.ByteBuffer

object NativeTracker {

    init {
//        System.loadLibrary("native-lib")
    }

    external fun processFrame(
        rgbaMatAddr: Long,
        width: Int,
        height: Int
    ): FloatArray?

    external fun detectEdges(
        y: ByteBuffer,
        u: ByteBuffer,
        v: ByteBuffer,
        width: Int,
        height: Int,
        yStride: Int,
        uvStride: Int,
        uvPixelStride: Int
    ): IntArray
}