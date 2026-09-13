package de.tudarmstadt.physics.trackingplot.ui.setup.tracker

import android.graphics.Bitmap
import android.graphics.Path
import android.graphics.Rect
import android.graphics.RectF
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import de.tudarmstadt.physics.trackingplot.R
import de.tudarmstadt.physics.trackingplot.geometry.FrameRotation
import de.tudarmstadt.physics.trackingplot.ui.common.FrameMapper
import de.tudarmstadt.physics.trackingplot.ui.common.rememberFrameRotation
import kotlin.math.roundToInt

private val MagnifierRadius = 80.dp
private const val MAGNIFIER_ZOOM = 2f
private const val DRAG_SENSITIVITY = 0.35f

/** Vertical gap between the fingertip and the bottom edge of the loupe drawn above it. */
private val LoupeOffsetAboveFinger = 24.dp

/**
 * Full-screen colour picker over a frozen camera snapshot (in canonical, portrait orientation).
 * Tap or drag to move a magnifying loupe and sample the pixel underneath; "Use colour" confirms
 * the last sampled colour. In landscape the displayed image (and the touch mapping) are rotated to
 * match the current display orientation, exactly like the live viewfinder.
 */
@Composable
fun ColorPickerOverlay(bitmap: Bitmap, onColorSelected: (Color) -> Unit, onCancel: () -> Unit) {
    val rotation = rememberFrameRotation()
    val imageBitmap = remember(bitmap) { bitmap.asImageBitmap() }
    val density = LocalDensity.current

    // Raw pointer position (moves 1:1 with the finger; positions the loupe above it) and the
    // sampled point (jumps to a tap, or trails the finger by dragAmount * DRAG_SENSITIVITY while
    // dragging; positions the ring and the zoomed content inside the loupe).
    var touchPosition by remember { mutableStateOf<Offset?>(null) }
    var pickPosition by remember { mutableStateOf<Offset?>(null) }
    var selectedColor by remember { mutableStateOf<Color?>(null) }

    Box(modifier = Modifier.fillMaxSize().safeDrawingPadding().background(Color.Black)) {
        BoxWithConstraints(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            val aspect = rotation.displayWidthOverHeight
            val useFullWidth = maxWidth / maxHeight <= aspect
            val outerWidth = if (useFullWidth) maxWidth else maxHeight * aspect
            val outerHeight = if (useFullWidth) maxWidth / aspect else maxHeight
            val canonicalWidth = if (rotation.swapsAxes) outerHeight else outerWidth
            val canonicalHeight = if (rotation.swapsAxes) outerWidth else outerHeight

            val mapper = remember(rotation, outerWidth, outerHeight, density) {
                FrameMapper(rotation, with(density) { Size(outerWidth.toPx(), outerHeight.toPx()) })
            }

            // Bitmap pixels per displayed pixel: the snapshot's resolution is generally not the
            // same as the on-screen box, so sampling sizes (the magnifier crop) must be scaled by
            // this factor to represent a consistent real zoom level.
            val bitmapPxPerDisplayPx = remember(bitmap, canonicalWidth, density) {
                bitmap.width / with(density) { canonicalWidth.toPx() }
            }

            fun pickColorAt(position: Offset) {
                val canonical = mapper.toCanonical(position)
                val x = (canonical.x * bitmap.width).toInt().coerceIn(0, bitmap.width - 1)
                val y = (canonical.y * bitmap.height).toInt().coerceIn(0, bitmap.height - 1)
                selectedColor = Color(bitmap.getPixel(x, y))
            }

            Box(modifier = Modifier.width(outerWidth).height(outerHeight)) {
                Image(
                    bitmap = imageBitmap,
                    contentDescription = null,
                    contentScale = ContentScale.Fit,
                    modifier = Modifier
                        .align(Alignment.Center)
                        .width(canonicalWidth)
                        .height(canonicalHeight)
                        .graphicsLayer { rotationZ = rotation.degrees.toFloat() }
                )

                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .pointerInput(mapper) {
                            detectTapGestures { offset ->
                                touchPosition = offset
                                pickPosition = offset
                                pickColorAt(offset)
                            }
                        }
                        .pointerInput(mapper) {
                            val boxSize = size
                            detectDragGestures(
                                onDragStart = { offset ->
                                    touchPosition = offset
                                    pickPosition = offset
                                    pickColorAt(offset)
                                }
                            ) { change, dragAmount ->
                                change.consume()
                                touchPosition = change.position
                                val current = pickPosition ?: change.position
                                val next = Offset(
                                    (current.x + dragAmount.x * DRAG_SENSITIVITY)
                                        .coerceIn(0f, boxSize.width.toFloat()),
                                    (current.y + dragAmount.y * DRAG_SENSITIVITY)
                                        .coerceIn(0f, boxSize.height.toFloat())
                                )
                                pickPosition = next
                                pickColorAt(next)
                            }
                        }
                )

                val touch = touchPosition
                val pick = pickPosition
                if (touch != null && pick != null) {
                    ColorMagnifierLoupe(
                        bitmap = bitmap,
                        mapper = mapper,
                        rotation = rotation,
                        touchPosition = touch,
                        pickPosition = pick,
                        color = selectedColor ?: Color.Transparent,
                        bitmapPxPerDisplayPx = bitmapPxPerDisplayPx
                    )
                }
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(CircleShape)
                    .background(selectedColor ?: Color.Transparent)
                    .border(width = 2.dp, color = Color.White, shape = CircleShape)
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                TextButton(onClick = onCancel) {
                    Text(text = stringResource(R.string.action_cancel), color = Color.White)
                }
                Button(onClick = { selectedColor?.let(onColorSelected) }, enabled = selectedColor != null) {
                    Text(stringResource(R.string.setup_color_picker_use_color))
                }
            }
        }
    }
}

@Composable
private fun BoxScope.ColorMagnifierLoupe(
    bitmap: Bitmap,
    mapper: FrameMapper,
    rotation: FrameRotation,
    touchPosition: Offset,
    pickPosition: Offset,
    color: Color,
    bitmapPxPerDisplayPx: Float
) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val radiusPx = MagnifierRadius.toPx()
        val loupeOffsetPx = LoupeOffsetAboveFinger.toPx()
        val canonical = mapper.toCanonical(pickPosition)
        val bitmapX = canonical.x * bitmap.width
        val bitmapY = canonical.y * bitmap.height
        // Half-size of the sampled bitmap region: a `radiusPx / MAGNIFIER_ZOOM` patch of the
        // displayed image, converted from display pixels to bitmap pixels.
        val srcHalf = (radiusPx / MAGNIFIER_ZOOM) * bitmapPxPerDisplayPx

        val srcRect = Rect(
            (bitmapX - srcHalf).roundToInt(),
            (bitmapY - srcHalf).roundToInt(),
            (bitmapX + srcHalf).roundToInt(),
            (bitmapY + srcHalf).roundToInt()
        )

        // The loupe sits above the fingertip so the sampled point stays visible; if there isn't
        // room above (finger near the top of the picker area), it clamps to the top edge instead
        // of jumping below the finger. It always stays fully inside the picker area horizontally.
        val lensCenter = Offset(
            touchPosition.x.coerceIn(radiusPx, (size.width - radiusPx).coerceAtLeast(radiusPx)),
            (touchPosition.y - (radiusPx + loupeOffsetPx))
                .coerceIn(radiusPx, (size.height - radiusPx).coerceAtLeast(radiusPx))
        )
        val dstRect = RectF(
            lensCenter.x - radiusPx,
            lensCenter.y - radiusPx,
            lensCenter.x + radiusPx,
            lensCenter.y + radiusPx
        )

        drawIntoCanvas { canvas ->
            val native = canvas.nativeCanvas
            native.save()
            native.clipPath(
                Path().apply { addCircle(lensCenter.x, lensCenter.y, radiusPx, Path.Direction.CCW) }
            )
            native.rotate(rotation.degrees.toFloat(), lensCenter.x, lensCenter.y)
            native.drawBitmap(bitmap, srcRect, dstRect, null)
            native.restore()
        }

        drawCircle(color = Color.White, radius = radiusPx, center = lensCenter, style = Stroke(width = 3.dp.toPx()))
        drawCircle(color = color, radius = 8.dp.toPx(), center = lensCenter)
        drawCircle(color = Color.White, radius = 6.dp.toPx(), center = pickPosition, style = Stroke(width = 2.dp.toPx()))
    }
}
