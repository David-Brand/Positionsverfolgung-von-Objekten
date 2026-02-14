package de.tudarmstadt.physics.trackingplot.ui.setup.tracker

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.IntSize

fun Bitmap.getColorAt(offset: Offset, layoutSize: IntSize): Color {
    if (layoutSize == IntSize.Zero) return Color.White

    val scale = maxOf(
        layoutSize.width.toFloat() / width,
        layoutSize.height.toFloat() / height
    )
    val scaledW = width * scale
    val scaledH = height * scale

    val cropX = (scaledW - layoutSize.width) / 2
    val cropY = (scaledH - layoutSize.height) / 2

    val bitmapX = ((offset.x + cropX) / scale).toInt().coerceIn(0, width - 1)
    val bitmapY = ((offset.y + cropY) / scale).toInt().coerceIn(0, height - 1)

    return Color(getPixel(bitmapX, bitmapY))
}

fun ImageProxy.toBitmap2(): Bitmap {
    val buffer = planes[0].buffer
    buffer.rewind()
    val bytes = ByteArray(buffer.capacity())
    buffer.get(bytes)

    val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)

    // Rotate if needed (CameraX gives you the rotation)
    val matrix = Matrix().apply {
        postRotate(imageInfo.rotationDegrees.toFloat())
    }
    return Bitmap.createBitmap(
        bitmap,
        0, 0,
        bitmap.width, bitmap.height,
        matrix, true
    )
}