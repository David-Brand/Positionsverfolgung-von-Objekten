package de.tudarmstadt.physics.trackingplot.camera

import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.graphics.Rect
import androidx.camera.core.ImageProxy
import de.tudarmstadt.physics.trackingplot.model.Roi
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import kotlin.math.roundToInt

/**
 * Turns CameraX analysis frames into the canonical frame (see FrameGeometry): the RGBA buffer is
 * wrapped without copying, cropped to the ViewPort crop rectangle and rotated into portrait
 * orientation into a reused Mat. Not thread-safe; use from the analysis thread only.
 */
class FrameProcessor : AutoCloseable {

    private val canonical = Mat()

    /**
     * Returns the canonical RGBA frame (CV_8UC4), valid until the next call, or null when the
     * frame cannot be used (CameraX delivers the raw YUV image if its RGBA conversion failed).
     * Must be called before [ImageProxy.close].
     */
    fun toCanonicalFrame(image: ImageProxy): Mat? {
        if (image.format != PixelFormat.RGBA_8888) return null
        val plane = image.planes.singleOrNull() ?: return null
        if (plane.pixelStride != 4) return null

        // The row stride can be larger than width * 4; pass it as the Mat step.
        val wrapped = Mat(image.height, image.width, CvType.CV_8UC4, plane.buffer, plane.rowStride.toLong())
        try {
            val crop = Rect(image.cropRect).apply {
                if (!intersect(0, 0, image.width, image.height) || isEmpty) {
                    set(0, 0, image.width, image.height)
                }
            }
            val sub = wrapped.submat(crop.top, crop.bottom, crop.left, crop.right)
            try {
                when (image.imageInfo.rotationDegrees) {
                    90 -> Core.rotate(sub, canonical, Core.ROTATE_90_CLOCKWISE)
                    180 -> Core.rotate(sub, canonical, Core.ROTATE_180)
                    270 -> Core.rotate(sub, canonical, Core.ROTATE_90_COUNTERCLOCKWISE)
                    else -> sub.copyTo(canonical)
                }
            } finally {
                sub.release()
            }
        } finally {
            wrapped.release()
        }
        return canonical
    }

    /** Copies an RGBA frame into a new bitmap. */
    fun toBitmap(frame: Mat): Bitmap {
        val bitmap = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(frame, bitmap)
        return bitmap
    }

    override fun close() {
        canonical.release()
    }
}

/** Pixel rectangle of this region inside a frame of the given size (canonical orientation). */
fun Roi.toPixelRect(frameWidth: Int, frameHeight: Int): org.opencv.core.Rect {
    val l = (left * frameWidth).roundToInt().coerceIn(0, frameWidth)
    val r = (right * frameWidth).roundToInt().coerceIn(0, frameWidth)
    val t = (top * frameHeight).roundToInt().coerceIn(0, frameHeight)
    val b = (bottom * frameHeight).roundToInt().coerceIn(0, frameHeight)
    return org.opencv.core.Rect(l, t, r - l, b - t)
}
