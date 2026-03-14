package de.tudarmstadt.physics.trackingplot.ui.setup.tracker

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import kotlin.math.min

@Composable
fun ColorPickerOverlay(
    bitmap: Bitmap,
    modifier: Modifier = Modifier,
    magnifierRadius: Dp = 80.dp,
    zoom: Float = 2f,
    dragSensitivity: Float = 0.35f,
    onColorChanged: (Color) -> Unit,
    onColorSelected: (Color) -> Unit,
    onCancel: () -> Unit
) {
    val imageBitmap = remember(bitmap) { bitmap.asImageBitmap() }

    var pickerPosition by remember { mutableStateOf(Offset.Zero) }
    var imageSize by remember { mutableStateOf(IntSize.Zero) }
    var selectedColor by remember { mutableStateOf(Color.Transparent) }

    Box(
        modifier = modifier
            .background(Color.Black)
            .onSizeChanged { imageSize = it }
            .pointerInput(bitmap) {
                detectTapGestures { offset ->
                    // Slow drag movement
                    pickerPosition = offset

                    pickerPosition = Offset(
                        pickerPosition.x.coerceIn(0f, imageSize.width.toFloat()),
                        pickerPosition.y.coerceIn(0f, imageSize.height.toFloat())
                    )

                    val bitmapOffset = mapTouchToBitmap(
                        touch = pickerPosition,
                        imageSize = imageSize,
                        bitmap = bitmap
                    )

                    val x = bitmapOffset.x
                        .toInt()
                        .coerceIn(0, bitmap.width - 1)

                    val y = bitmapOffset.y
                        .toInt()
                        .coerceIn(0, bitmap.height - 1)

//                    val size = this.size
//                    val normalized = Offset(
//                        (offset.x / size.width).coerceIn(0.0f, 1.0f),
//                        (offset.y / size.height).coerceIn(0.0f, 1.0f)
//                    )
//                    val size = imageSize
//                    val normalized = Offset(
//                        (bitmapOffset.x / size.width).coerceIn(0.0f, 1.0f),
//                        (bitmapOffset.y / size.height).coerceIn(0.0f, 1.0f)
//                    )
//                    println(normalized)

                    selectedColor = Color(bitmap.getPixel(x, y))
                    onColorChanged(selectedColor)
                }
            }
            .pointerInput(bitmap) {
                detectDragGestures(
                    onDragStart = { offset ->
                        pickerPosition = offset
                    }
                ) { change, dragAmount ->

                    change.consume()

                    // Slow drag movement
                    pickerPosition += dragAmount * dragSensitivity

                    pickerPosition = Offset(
                        pickerPosition.x.coerceIn(0f, imageSize.width.toFloat()),
                        pickerPosition.y.coerceIn(0f, imageSize.height.toFloat())
                    )

                    val bitmapOffset = mapTouchToBitmap(
                        touch = pickerPosition,
                        imageSize = imageSize,
                        bitmap = bitmap
                    )

                    val x = bitmapOffset.x
                        .toInt()
                        .coerceIn(0, bitmap.width - 1)

                    val y = bitmapOffset.y
                        .toInt()
                        .coerceIn(0, bitmap.height - 1)

                    selectedColor = Color(bitmap.getPixel(x, y))
                    onColorChanged(selectedColor)
                }
            }
    ) {

        // Image
        Image(
            bitmap = imageBitmap,
            contentDescription = null,
            contentScale = ContentScale.Fit,
            modifier = Modifier.fillMaxSize()
        )

        // Magnifier
        Canvas(modifier = Modifier.fillMaxSize()) {

            if (pickerPosition == Offset.Zero) return@Canvas

            val radiusPx = magnifierRadius.toPx()
            val srcSize = radiusPx / zoom

            val bitmapOffset = mapTouchToBitmap(
                touch = pickerPosition,
                imageSize = imageSize,
                bitmap = bitmap
            )

            val srcRect = android.graphics.Rect(
                (bitmapOffset.x - srcSize).toInt(),
                (bitmapOffset.y - srcSize).toInt(),
                (bitmapOffset.x + srcSize).toInt(),
                (bitmapOffset.y + srcSize).toInt()
            )

            val dstRect = android.graphics.RectF(
                pickerPosition.x - radiusPx,
                pickerPosition.y - radiusPx,
                pickerPosition.x + radiusPx,
                pickerPosition.y + radiusPx
            )

            drawIntoCanvas { canvas ->
                canvas.nativeCanvas.save()
                canvas.nativeCanvas.clipPath(
                    android.graphics.Path().apply {
                        addCircle(
                            pickerPosition.x,
                            pickerPosition.y,
                            radiusPx,
                            android.graphics.Path.Direction.CCW
                        )
                    }
                )

                canvas.nativeCanvas.drawBitmap(
                    bitmap,
                    srcRect,
                    dstRect,
                    null
                )

                canvas.nativeCanvas.restore()
            }

            // Outer border
            drawCircle(
                color = Color.White,
                radius = radiusPx,
                center = pickerPosition,
                style = Stroke(width = 4.dp.toPx())
            )

            // Center dot
            drawCircle(
                color = selectedColor,
                radius = 12.dp.toPx(),
                center = pickerPosition
            )
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
                .align(Alignment.TopCenter),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            // Selected color preview
            Box(
                modifier = Modifier
                    .size(56.dp)
                    .background(selectedColor, CircleShape)
                    .border(2.dp, Color.White, CircleShape)
            )

            Row {
                TextButton(onClick = onCancel) { Text("Cancel") }
                Button(onClick = { onColorSelected(selectedColor) }) { Text("Use Color") }
            }
        }
    }
}

private fun mapTouchToBitmap(
    touch: Offset,
    imageSize: IntSize,
    bitmap: Bitmap
): Offset {

    val imageWidth = imageSize.width.toFloat()
    val imageHeight = imageSize.height.toFloat()

    val bitmapWidth = bitmap.width.toFloat()
    val bitmapHeight = bitmap.height.toFloat()

    val scale = min(
        imageWidth / bitmapWidth,
        imageHeight / bitmapHeight
    )

    val scaledWidth = bitmapWidth * scale
    val scaledHeight = bitmapHeight * scale

    val offsetX = (imageWidth - scaledWidth) / 2f
    val offsetY = (imageHeight - scaledHeight) / 2f

    val x = ((touch.x - offsetX) / scale)
    val y = ((touch.y - offsetY) / scale)

    return Offset(x, y)
}
