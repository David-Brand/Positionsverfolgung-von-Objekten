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
import androidx.compose.foundation.magnifier
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
import androidx.compose.ui.geometry.isSpecified
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import kotlin.math.min

//TODO VERSION 1
//@Composable
//fun ColorPickerOverlay(
//    bitmap: Bitmap,
//    onColorSelected: (Color) -> Unit,
//    onCancel: () -> Unit
//) {
//    var pickerPosition by remember { mutableStateOf(Offset.Zero) }
//    var layoutSize by remember { mutableStateOf(IntSize.Zero) }
//    var currentColor by remember { mutableStateOf(Color.White) }
//
//    val imageBitmap = remember(bitmap) { bitmap.asImageBitmap() }
//
//    Box(modifier = Modifier.fillMaxSize()) {
//        // The frozen image with built-in magnifier lens
//        Image(
//            bitmap = imageBitmap,
//            contentDescription = null,
//            modifier = Modifier
//                .fillMaxSize()
//                .onGloballyPositioned { layoutSize = it.size }
//                .magnifier(
//                    sourceCenter = { pickerPosition },           // what to magnify
//                    magnifierCenter = { pickerPosition + Offset(60f, -180f) }, // lens position (above the picker)
//                    zoom = 7f,
//                    size = DpSize(140.dp, 140.dp),
//                    cornerRadius = 70.dp,
//                    elevation = 8.dp,
//                    clip = true
////                    clippingEnabled = true
//                ),
//            contentScale = ContentScale.Fit   // matches typical camera preview
//        )
//
//        // Drag/tap layer (transparent, covers the whole image)
//        Box(
//            modifier = Modifier
//                .fillMaxSize()
//                .pointerInput(Unit) {
//                    // Tap to jump
//                    detectTapGestures { offset ->
//                        pickerPosition = offset.coerceIn(layoutSize)
//                        currentColor = bitmap.getColorAt(offset, layoutSize)
////                        onColorSelected(currentColor)
//                    }
//                }
//                .pointerInput(Unit) {
//                    // Drag with reduced speed
//                    detectDragGestures(
//                        onDragStart = { offset ->
//                            pickerPosition = offset.coerceIn(layoutSize)
//                            currentColor = bitmap.getColorAt(offset, layoutSize)
////                            onColorSelected(currentColor)
//                        },
//                        onDrag = { change, dragAmount ->
//                            change.consume()
//                            val slowed = dragAmount * 0.35f   // ← tune this (0.2f–0.5f feels great)
//                            pickerPosition = (pickerPosition + slowed).coerceIn(layoutSize)
//                            currentColor = bitmap.getColorAt(pickerPosition, layoutSize)
////                            onColorSelected(currentColor)
//                        }
//                    )
//                }
//        ) {
//            // Visual picker (white ring + inner color dot)
//            Canvas(modifier = Modifier.fillMaxSize()) {
//                drawCircle(
//                    color = Color.White,
//                    radius = 24f,
//                    center = pickerPosition,
//                    style = Stroke(width = 5f)
//                )
//                drawCircle(
//                    color = currentColor,
//                    radius = 16f,
//                    center = pickerPosition
//                )
//            }
//        }
//
//        // Top bar with selected color preview + buttons
//        Row(
//            modifier = Modifier
//                .fillMaxWidth()
//                .padding(16.dp)
//                .align(Alignment.TopCenter),
//            horizontalArrangement = Arrangement.SpaceBetween
//        ) {
//            // Selected color preview
//            Box(
//                modifier = Modifier
//                    .size(56.dp)
//                    .background(currentColor, CircleShape)
//                    .border(2.dp, Color.White, CircleShape)
//            )
//
//            Row {
//                TextButton(onClick = onCancel) { Text("Cancel") }
//                Button(onClick = { onColorSelected(currentColor) }) { Text("Use Color") }
//            }
//        }
//    }
//}
//
//// Small helper to clamp offset
//private fun Offset.coerceIn(size: IntSize): Offset =
//    Offset(
//        x = x.coerceIn(0f, size.width.toFloat()),
//        y = y.coerceIn(0f, size.height.toFloat())
//    )

//TODO VERSION 2
//@Composable
//fun ColorPickerOverlay(
//    bitmap: Bitmap,
//    onColorSelected: (Color) -> Unit,
//    onCancel: () -> Unit
//) {
//    var pickerPosition by remember { mutableStateOf(Offset.Zero) }           // slowed for precision
//    var magnifierPosition by remember { mutableStateOf(Offset.Unspecified) } // raw/fast for lens
//    var layoutSize by remember { mutableStateOf(IntSize.Zero) }
//    var currentColor by remember { mutableStateOf(Color.White) }
//    val imageBitmap = remember(bitmap) { bitmap.asImageBitmap() }
//
//    Box(modifier = Modifier.fillMaxSize()) {
//        Image(
//            bitmap = imageBitmap,
//            contentDescription = null,
//            modifier = Modifier
//                .fillMaxSize()
//                .onGloballyPositioned { layoutSize = it.size }
//                .pointerInput(Unit) {
//                    // Optional: center on first appearance
//                    if (pickerPosition == Offset.Zero && magnifierPosition == Offset.Unspecified) {
//                        val center = Offset(layoutSize.width / 2f, layoutSize.height / 2f)
//                        pickerPosition = center
//                        magnifierPosition = center
//                        currentColor = bitmap.getColorAt(center, layoutSize)
//                    }
//                }
//                .magnifier(
//                    sourceCenter = { magnifierPosition },
//                    magnifierCenter = {
//                        if (magnifierPosition.isSpecified) {
//                            magnifierPosition + Offset(60f, -180f)
//                        } else {
//                            Offset.Unspecified
//                        }
//                    },
//                    zoom = 7f,
//                    size = DpSize(140.dp, 140.dp),
//                    cornerRadius = 70.dp,
//                    elevation = 8.dp,
//                    clip = true
//                ),
//            contentScale = ContentScale.Fit
//        )
//
//        Box(
//            modifier = Modifier
//                .fillMaxSize()
//                .pointerInput(Unit) {
//                    detectTapGestures { offset ->
//                        val pos = offset.coerceIn(layoutSize)
//                        pickerPosition = pos
//                        magnifierPosition = pos   // snap lens too
//                        currentColor = bitmap.getColorAt(pos, layoutSize)
//                    }
//                }
//                .pointerInput(Unit) {
//                    detectDragGestures(
//                        onDragStart = { offset ->
//                            val pos = offset.coerceIn(layoutSize)
//                            pickerPosition = pos
//                            magnifierPosition = pos
//                            currentColor = bitmap.getColorAt(pos, layoutSize)
//                        },
//                        onDrag = { change, dragAmount ->
//                            change.consume()
//
//                            // Picker moves slowly (precision)
//                            val slowed = dragAmount * 0.35f
//                            pickerPosition = (pickerPosition + slowed).coerceIn(layoutSize)
//                            currentColor = bitmap.getColorAt(pickerPosition, layoutSize)
//
//                            // Lens follows finger directly (no lag feel)
//                            magnifierPosition = (magnifierPosition + dragAmount).coerceIn(layoutSize)
//                        },
//                        onDragEnd = {
//                            // Optional: keep lens visible or hide
//                            // magnifierPosition = Offset.Unspecified
//                        },
//                        onDragCancel = {
//                            magnifierPosition = Offset.Unspecified
//                        }
//                    )
//                }
//        ) {
//            Canvas(modifier = Modifier.fillMaxSize()) {
//                if (pickerPosition != Offset.Zero) {
//                    drawCircle(
//                        color = Color.White,
//                        radius = 24f,
//                        center = pickerPosition,
//                        style = Stroke(width = 5f)
//                    )
//                    drawCircle(
//                        color = currentColor,
//                        radius = 16f,
//                        center = pickerPosition
//                    )
//                }
//            }
//        }
//
//        // Top bar (unchanged)
//        Row(
//            modifier = Modifier
//                .fillMaxWidth()
//                .padding(16.dp)
//                .align(Alignment.TopCenter),
//            horizontalArrangement = Arrangement.SpaceBetween
//        ) {
//            Box(
//                modifier = Modifier
//                    .size(56.dp)
//                    .background(currentColor, CircleShape)
//                    .border(2.dp, Color.White, CircleShape)
//            )
//
//            Row {
//                TextButton(onClick = onCancel) { Text("Cancel") }
//                Button(onClick = { onColorSelected(currentColor) }) { Text("Use Color") }
//            }
//        }
//    }
//}
//
//// Your coerceIn helper (unchanged)
//private fun Offset.coerceIn(size: IntSize): Offset =
//    Offset(
//        x = x.coerceIn(0f, size.width.toFloat()),
//        y = y.coerceIn(0f, size.height.toFloat())
//    )

@Composable
fun BitmapColorPicker(
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
//        Canvas(modifier = Modifier.matchParentSize()) {

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
