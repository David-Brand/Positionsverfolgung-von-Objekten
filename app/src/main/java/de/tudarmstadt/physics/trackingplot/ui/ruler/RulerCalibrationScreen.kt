package de.tudarmstadt.physics.trackingplot.ui.ruler

import android.graphics.PointF
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.drawText
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import de.tudarmstadt.physics.trackingplot.R
import de.tudarmstadt.physics.trackingplot.ui.NewCameraPreview

@Composable
fun RulerCalibrationScreen(
    viewModel: CameraViewModel = viewModel(),
    onDone: (PointF, PointF) -> Unit
) {
    NewCameraPreview()
    return

    var currentPoints by remember { mutableStateOf(listOf<PointF>()) }

    CameraWithOverlay(
        onPreviewSizeChanged = { viewModel.previewSize = it }
    ) { previewSize ->

        Box(modifier = Modifier.fillMaxSize()) {
            Canvas(modifier = Modifier
                .fillMaxSize()
                .pointerInput(Unit) {
                    detectTapGestures { offset ->
                        val point = PointF(offset.x, offset.y)

                        if (currentPoints.size < 2) {
                            currentPoints = currentPoints + point
                        }
                    }
                    // You can also add long press → clear, etc.
                }
            ) {
                currentPoints.forEachIndexed { idx, pt ->
                    // Draw dot
                    drawCircle(
                        color = Color.Red,
                        radius = 16.dp.toPx(),
                        center = Offset(pt.x, pt.y)
                    )

                    // Optional label
//                    if (idx == 0) drawText("1", pt.x + 24, pt.y, textPaint)
//                    if (idx == 1) drawText("2", pt.x + 24, pt.y, textPaint)
                }

                if (currentPoints.size == 2) {
                    drawLine(
                        color = Color.Yellow,
                        start = Offset(currentPoints[0].x, currentPoints[0].y),
                        end = Offset(currentPoints[1].x, currentPoints[1].y),
                        strokeWidth = 4.dp.toPx()
                    )
                }
            }

            // Draggable points
            currentPoints.forEachIndexed { index, point ->
                DraggablePoint(
                    initialPosition = point,
                    onPositionChanged = { newPos ->
                        val newList = currentPoints.toMutableList()
                        newList[index] = newPos
                        currentPoints = newList
                    }
                )
            }

            if (currentPoints.size == 2) {
                Button(
                    onClick = {
                        viewModel.point1 = currentPoints[0]
                        viewModel.point2 = currentPoints[1]
                        onDone(currentPoints[0], currentPoints[1])
                    },
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(24.dp)
                ) {
                    Text("Confirm & Continue")
                }
            }
        }
    }
}

@Composable
fun DraggablePoint(
    initialPosition: PointF,
    onPositionChanged: (PointF) -> Unit
) {
    var offset by remember { mutableStateOf(Offset(initialPosition.x, initialPosition.y)) }

    Box(
        modifier = Modifier
            .offset { IntOffset(offset.x.toInt(), offset.y.toInt()) }
            .size(48.dp)
            .pointerInput(Unit) {
                detectDragGestures { change, dragAmount ->
                    change.consume()
                    offset = Offset(
                        (offset.x + dragAmount.x).coerceIn(0f, size.width.toFloat()),
                        (offset.y + dragAmount.y).coerceIn(0f, size.height.toFloat())
                    )
                    onPositionChanged(PointF(offset.x, offset.y))
                }
            }
    ) {
        Icon(
            painter = painterResource(id = R.drawable.ic_launcher_foreground),
//            imageVector = Icons.Default.RadioButtonChecked,
            contentDescription = null,
            tint = Color.Red,
            modifier = Modifier
                .size(32.dp)
                .align(Alignment.Center)
        )
    }
}