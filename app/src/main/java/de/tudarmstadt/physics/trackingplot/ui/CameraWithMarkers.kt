package de.tudarmstadt.physics.trackingplot.ui

import android.graphics.PointF
import androidx.camera.compose.CameraXViewfinder
import androidx.camera.core.SurfaceRequest
import androidx.camera.viewfinder.compose.MutableCoordinateTransformer
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.ui.graphics.Canvas
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt

@Composable
fun CameraWithMarkersOld(
    surfaceRequest: SurfaceRequest,
    onDone: (List<Offset>) -> Unit   // or List<Offset>, whatever you prefer
) {
//    val points = remember { mutableStateListOf<Offset>() }           // UI-space points
//    val transformer = remember { MutableCoordinateTransformer() }
    val points = remember { mutableStateListOf<Offset>() }  // max 2
    val transformer = remember { MutableCoordinateTransformer() }

    var draggedIndex by remember { mutableStateOf(-1) }  // which point is being dragged (-1 = none)

    Box(
        modifier = Modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                detectTapGestures { tapOffset ->
                    if (points.size < 2) {
                        // Add new point centered on tap
                        points.add(tapOffset)
                    } else {
                        // Optional: replace closest if you want to allow moving existing
                        // but for now we just limit to 2 taps
                    }
                }

                detectDragGestures(
                    onDragStart = { startOffset ->
                        println("Drag STARTED at $startOffset")
                        // your closest point logic
                    },
                    onDragEnd = {
                        println("Drag ENDED")
                    },
                    onDragCancel = {
                        println("Drag CANCELLED")
                    },
                    onDrag = { change, dragAmount ->
                        println("Dragging: $dragAmount")
                        // your move logic
                    }
//                    onDragStart = { startOffset ->
//                        // Find the closest point to start position (within reasonable distance)
//                        val closestIndex = points.indexOfFirst { point ->
//                            (point - startOffset).getDistance() < 60.dp.toPx()  // hit radius ~60dp
//                        }
//
//                        if (closestIndex != -1) {
//                            draggedIndex = closestIndex
//                        }
//                    },
//                    onDragEnd = {
//                        draggedIndex = -1
//                    },
//                    onDragCancel = {
//                        draggedIndex = -1
//                    },
//                    onDrag = { change, dragAmount ->
//                        change.consume()
//
//                        if (draggedIndex >= 0 && draggedIndex < points.size) {
//                            val old = points[draggedIndex]
//                            var newX = old.x + dragAmount.x
//                            var newY = old.y + dragAmount.y
//
//                            // Optional: keep inside preview bounds
//                            newX = newX.coerceIn(0f, size.width.toFloat())
//                            newY = newY.coerceIn(0f, size.height.toFloat())
//
//                            points[draggedIndex] = Offset(newX, newY)
//                        }
//                    }
                )
            }
//            .pointerInput(Unit) {
//                detectTapGestures { offset ->
//                    // Only add if inside viewfinder bounds (optional extra check)
//                    if (offset.x in 0f..size.width.toFloat() && offset.y in 0f..size.height.toFloat()) {
//                        points.add(offset)
//                    }
//                }
//
//                detectDragGestures { change, dragAmount ->
//                    change.consume()
//
//                    val newPoints = points.map { pt ->
//                        var newX = pt.x + dragAmount.x
//                        var newY = pt.y + dragAmount.y
//
//                        // Optional: clamp to viewfinder area
//                        newX = newX.coerceIn(0f, size.width.toFloat())
//                        newY = newY.coerceIn(0f, size.height.toFloat())
//
//                        Offset(newX, newY)
//                    }
//                    points.clear()
//                    points.addAll(newPoints)
//                }
//            }
    ) {
        CameraXViewfinder(
            surfaceRequest = surfaceRequest,
            coordinateTransformer = transformer,
//            modifier = Modifier.matchParentSize(),
//            contentScale = ContentScale.Fit
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Fit,     // letterbox / preserve ratio
            alignment = Alignment.Center
        )

//        // Draw the dots (overlay)
//        points.forEach { pt ->
//            Canvas(
//                modifier = Modifier
//                    .offset { IntOffset(pt.x.toInt(), pt.y.toInt()) }
//                    .size(32.dp)  // diameter
//            ) {
//                drawCircle(
//                    color = Color.Red.copy(alpha = 0.7f),
//                    radius = size.minDimension / 2,
//                    style = Stroke(width = 4.dp.toPx())
//                )
//            }
//        }
        // Draw the 1–2 points as centered circles
        points.forEachIndexed { index, center ->
            val isBeingDragged = (index == draggedIndex)

            Box(
                modifier = Modifier
                    .offset {
                        // Center the 32dp dot on the coordinate
                        IntOffset(
                            (center.x - 16.dp.toPx()).roundToInt(),
                            (center.y - 16.dp.toPx()).roundToInt()
                        )
                    }
                    .size(32.dp)
                    .clip(CircleShape)
                    .background(
                        if (isBeingDragged) Color.Yellow.copy(alpha = 0.8f)
                        else Color.Red.copy(alpha = 0.7f)
                    )
                    .border(3.dp, Color.White, CircleShape)  // optional white ring for visibility
            )
        }

        // Done button (example placement)
        if (points.size == 2) {
            Button(
                onClick = {
                    val imageCoords = points.map { uiOffset ->
                        with(transformer) { uiOffset.transform() }
                    }
                    onDone(imageCoords)
                },
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 32.dp)
            ) {
                Text("Done – Use These 2 Points")
            }
        }
//        // Done button somewhere in UI
//        Button(
//            onClick = {
//                val imagePoints = points.map { uiOffset ->
//                    with(transformer) {
//                        uiOffset.transform()   // → sensor / buffer coordinates
//                    }
//                }
//                onDone(imagePoints)
//            },
//            modifier = Modifier.align(Alignment.BottomCenter)
//        ) {
//            Text("Done")
//        }
    }
}

@Composable
fun CameraWithMarkers(
    surfaceRequest: SurfaceRequest,
    onDone: (List<Offset>) -> Unit
) {
    val points = remember { mutableStateListOf<Offset>() }  // max 2
    val transformer = remember { MutableCoordinateTransformer() }
    var draggedIndex by remember { mutableStateOf(-1) }

    Box(Modifier.fillMaxSize()) {

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
                        .background(if (isDragged) Color.Yellow.copy(alpha = 0.8f) else Color.Red.copy(alpha = 0.7f))
                        .border(3.dp, Color.White, CircleShape)
                )
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