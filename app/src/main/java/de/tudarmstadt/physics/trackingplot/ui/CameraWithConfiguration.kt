package de.tudarmstadt.physics.trackingplot.ui

import androidx.camera.compose.CameraXViewfinder
import androidx.camera.core.SurfaceRequest
import androidx.camera.viewfinder.compose.MutableCoordinateTransformer
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.focusable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import de.tudarmstadt.physics.trackingplot.ui.configuration.ColorTrackingConfigurator
import kotlin.math.roundToInt

@Composable
fun CameraWithConfiguration(
    surfaceRequest: SurfaceRequest,
    onDone: (List<Offset>) -> Unit
) {
    val points = remember { mutableStateListOf<Offset>() }  // max 2
    val transformer = remember { MutableCoordinateTransformer() }
    var draggedIndex by remember { mutableStateOf(-1) }

    Box(Modifier.fillMaxSize()) {

        ColorTrackingConfigurator() { }
        return

        // 1. Camera preview – NO gestures here
        CameraXViewfinder(
            surfaceRequest = surfaceRequest,
            coordinateTransformer = transformer,
            modifier = Modifier
                .matchParentSize()
                .focusable(false)           // helps a tiny bit sometimes
                .then(Modifier.pointerInput(Unit) { /* empty – prevents accidental consumption */ }),
            contentScale = ContentScale.Fit,
            alignment = Alignment.Center
        )

        // 2. Transparent overlay layer – THIS receives all touches
        Box(
            modifier = Modifier
                .matchParentSize()
                .background(Color.Transparent)  // crucial: must be transparent
                .pointerInput(Unit) {
                    detectTapGestures { tapOffset ->
                        println("Tap detected at $tapOffset")
                        if (points.size < 2) {
                            points.add(tapOffset)
                        }
                    }
                }
                .pointerInput(Unit) {
                    detectDragGestures(
                        onDragStart = { startOffset ->
                            println("Drag START at $startOffset")
                            val closestIndex = points.indexOfFirst { pt ->
                                (pt - startOffset).getDistance() < 60.dp.toPx()  // hit area
                            }
                            draggedIndex = closestIndex
                        },
                        onDrag = { change, dragAmount ->
                            change.consume()  // important
                            println("Drag delta: $dragAmount")

                            if (draggedIndex in points.indices) {
                                val old = points[draggedIndex]
                                var newX = old.x + dragAmount.x
                                var newY = old.y + dragAmount.y
                                newX = newX.coerceIn(0f, size.width.toFloat())
                                newY = newY.coerceIn(0f, size.height.toFloat())
                                points[draggedIndex] = Offset(newX, newY)
                            }
                        },
                        onDragEnd = {
                            println("Drag ended")
                            draggedIndex = -1
                        },
                        onDragCancel = {
                            draggedIndex = -1
                        }
                    )
                }
        ) {
            // Draw rectangle (when we have 2 points = opposite corners)
            if (points.size == 2) {
                val topLeft = Offset(
                    minOf(points[0].x, points[1].x),
                    minOf(points[0].y, points[1].y)
                )
                val bottomRight = Offset(
                    maxOf(points[0].x, points[1].x),
                    maxOf(points[0].y, points[1].y)
                )

                Canvas(modifier = Modifier.matchParentSize()) {
                    drawRect(
                        color = Color.Green.copy(alpha = 0.5f),
                        topLeft = topLeft,
                        size = Size(
                            width = bottomRight.x - topLeft.x,
                            height = bottomRight.y - topLeft.y
                        ),
                        style = Stroke(
                            width = 3.dp.toPx(),
                            cap = StrokeCap.Round,
                            join = StrokeJoin.Round
                        )
                    )
                }
            }

//            // 3. Draw your centered dots here (same as before)
//            points.forEachIndexed { index, center ->
//                val isDragged = index == draggedIndex
//                Box(
//                    modifier = Modifier
//                        .offset {
//                            IntOffset(
//                                x = (center.x - 16.dp.toPx()).roundToInt(),
//                                y = (center.y - 16.dp.toPx()).roundToInt()
//                            )
//                        }
//                        .size(32.dp)
//                        .clip(CircleShape)
//                        .background(if (isDragged) Color.Yellow.copy(alpha = 0.8f) else Color.Red.copy(alpha = 0.7f))
//                        .border(3.dp, Color.White, CircleShape)
//                )
//            }
            // 3. Draw your centered dots here (same as before)
            points.forEachIndexed { index, center ->
                val isDragged = index == draggedIndex
                Box(
                    modifier = Modifier
                        .offset {
                            IntOffset(
                                x = (center.x - 16.dp.toPx()).roundToInt(),
                                y = (center.y - 16.dp.toPx()).roundToInt()
                            )
                        }
                        .size(32.dp)
                        .clip(CircleShape)
                        .background(if (isDragged) Color.Yellow.copy(alpha = 0.3f) else Color.Red.copy(alpha = 0.2f))
                        .border(2.dp, Color.White, CircleShape)
                ) {
                    Canvas(
                        modifier = Modifier.size(32.dp)
                    ) {
                        val strokeWidth = (0.5).dp.toPx()
                        val halfWidth = size.width / 2
                        val halfHeight = size.height / 2

                        // Vertical line
                        drawLine(
                            color = Color.White,
                            start = Offset(halfWidth, 0f),
                            end = Offset(halfWidth, size.height),
                            strokeWidth = strokeWidth
                        )

                        // Horizontal line
                        drawLine(
                            color = Color.White,
                            start = Offset(0f, halfHeight),
                            end = Offset(size.width, halfHeight),
                            strokeWidth = strokeWidth
                        )
                    }
                }
            }
        }

        // Done button (only show when 2 points ready)
        if (points.size == 2) {
            Button(
                onClick = {
                    val imageCoords = points.map { ui ->
                        with(transformer) { ui.transform() }
                    }
                    onDone(imageCoords)
                },
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 48.dp)
            ) {
                Text("Done")
            }
        }
    }
}