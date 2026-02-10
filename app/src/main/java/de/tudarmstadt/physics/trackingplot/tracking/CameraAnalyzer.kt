package de.tudarmstadt.physics.trackingplot.tracking

import android.graphics.ImageFormat
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.compose.ui.geometry.Rect

class CameraAnalyzer(
    private val onResult: (Rect) -> Unit
) : ImageAnalysis.Analyzer {

    override fun analyze(image: ImageProxy) {
        try {
            image.image?.let {
                if (it.format == ImageFormat.YUV_420_888 && it.planes.size == 3) {
                    val rgbaMat = it.yuvToRgba()

                    val results = NativeTracker.processFrame(
                        rgbaMat.nativeObjAddr,
                        image.width,
                        image.height
                    )
                }
            }
        } catch (e: IllegalStateException) {
            e.printStackTrace()
        } finally {
            image.close()
        }
//        val planes = image.planes
//
//        image.image
//
//        val result = NativeTracker.detectEdges(
//            planes[0].buffer,
//            planes[1].buffer,
//            planes[2].buffer,
//            image.width,
//            image.height,
//            planes[0].rowStride,
//            planes[1].rowStride,
//            planes[1].pixelStride
//        )
//
//        if (result.size == 4) {
//            onResult(
//                Rect(
//                    result[0].toFloat(),
//                    result[1].toFloat(),
//                    (result[0] + result[2]).toFloat(),
//                    (result[1] + result[3]).toFloat()
//                )
//            )
//        }
    }
}