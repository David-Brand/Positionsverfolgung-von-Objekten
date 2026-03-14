package de.tudarmstadt.physics.trackingplot.ui.setup.bounding_box

import androidx.camera.compose.CameraXViewfinder
import androidx.camera.core.CameraSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.lifecycle.awaitInstance
import androidx.camera.viewfinder.compose.MutableCoordinateTransformer
import androidx.camera.viewfinder.core.ImplementationMode
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import de.tudarmstadt.physics.trackingplot.ui.setup.SetupViewModel
import kotlin.math.roundToInt

@Composable
fun BoundingBoxSetupScreen(
    back: () -> Unit,
    toNextStep: () -> Unit,
    skip: () -> Unit,
    setupViewModel: SetupViewModel
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val selector = CameraSelector.DEFAULT_BACK_CAMERA

    LaunchedEffect(Unit) {
        val provider = ProcessCameraProvider.awaitInstance(context)
        val useCaseGroup = setupViewModel.createUseCaseGroup()

        provider.unbindAll()
        provider.bindToLifecycle(lifecycleOwner, selector, useCaseGroup)
    }

    val surfaceRequest by setupViewModel.surfaceRequests.collectAsStateWithLifecycle()

    val uiPoints = setupViewModel.boundingUiPoints

    surfaceRequest?.let {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .safeDrawingPadding()
        ) {
            BoxWithConstraints(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                contentAlignment = Alignment.Center
            ) {
                val aspectRatio = 3 to 4
                val maxAspectRatio: Float = maxWidth / maxHeight
                val aspectRatioFloat: Float = aspectRatio.first.toFloat() / aspectRatio.second
                val shouldUseMaxWidth = maxAspectRatio <= aspectRatioFloat
                val width = if (shouldUseMaxWidth) maxWidth else maxHeight * aspectRatioFloat
                val height = if (!shouldUseMaxWidth) maxHeight else maxWidth / aspectRatioFloat

                Box(
                    modifier = Modifier
                        .width(width)
                        .height(height)
                ) {
                    val implementationMode = ImplementationMode.EXTERNAL

                    val coordinateTransformer = remember { MutableCoordinateTransformer() }
                    var draggedIndex by remember { mutableStateOf(-1) }
                    CameraXViewfinder(
                        modifier = Modifier
                            .fillMaxSize()
                            .pointerInput(Unit) {
                                detectTapGestures(
                                    onTap = { offset ->
                                        if (uiPoints.size < 2) {
                                            setupViewModel._boundingUiPoints.add(offset)

                                            if (size.width > 0 && size.height > 0) {
                                                val normalizedX = (offset.x / size.width).coerceIn(0f, 1f)
                                                val normalizedY = (offset.y / size.height).coerceIn(0f, 1f)

                                                println("Normalized: $normalizedX, $normalizedY")
                                                setupViewModel._boundingNormalizedPoints.add(
                                                    Offset(normalizedX, normalizedY)
                                                )
//                                                setupViewModel._boundingNormalizedPoints[draggedIndex] =
                                            }

//                                            with(coordinateTransformer) {
//                                                val surfaceCoords = offset.transform()
//                                                val surfaceResolution = it.resolution
//
//                                                val normalized = Offset(
//                                                    (surfaceCoords.x / surfaceResolution.width).coerceIn(0.0f, 1.0f),
//                                                    (surfaceCoords.y / surfaceResolution.height).coerceIn(0.0f, 1.0f)
//                                                )
//
//                                                setupViewModel._boundingNormalizedPoints.add(normalized)
//                                            }
                                        }
                                    }
                                )
                            }
                            .pointerInput(Unit) {
                                detectDragGestures(
                                    onDragStart = { startOffset ->
                                        val closestIndex = uiPoints.indexOfFirst { pt ->
                                            (pt - startOffset).getDistance() < 60.dp.toPx()
                                        }
                                        draggedIndex = closestIndex
                                    },
                                    onDrag = { change, dragAmount ->
                                        change.consume()

                                        if (draggedIndex in uiPoints.indices) {
                                            val old = uiPoints[draggedIndex]
                                            var newX = old.x + dragAmount.x
                                            var newY = old.y + dragAmount.y
                                            newX = newX.coerceIn(0f, size.width.toFloat())
                                            newY = newY.coerceIn(0f, size.height.toFloat())
                                            setupViewModel._boundingUiPoints[draggedIndex] = Offset(newX, newY)
                                        }
                                    },
                                    onDragEnd = {
                                        if (draggedIndex in uiPoints.indices) {
                                            val offset = uiPoints[draggedIndex]

                                            if (size.width > 0 && size.height > 0) {
                                                val normalizedX = (offset.x / size.width).coerceIn(0f, 1f)
                                                val normalizedY = (offset.y / size.height).coerceIn(0f, 1f)

                                                println("Normalized: $normalizedX, $normalizedY")
                                                setupViewModel._boundingNormalizedPoints[draggedIndex] =
                                                    Offset(normalizedX, normalizedY)
                                            }

//                                            with(coordinateTransformer) {
//                                                val surfaceCoords = offset.transform()
//                                                val surfaceResolution = it.resolution
//
//                                                val normalized = Offset(
//                                                    (surfaceCoords.x / surfaceResolution.width).coerceIn(0.0f, 1.0f),
//                                                    (surfaceCoords.y / surfaceResolution.height).coerceIn(0.0f, 1.0f)
//                                                )
//
//                                                setupViewModel._boundingNormalizedPoints[draggedIndex] = normalized
//                                            }
                                        }
                                        draggedIndex = -1
                                    },
                                    onDragCancel = {
                                        if (draggedIndex in uiPoints.indices) {
                                            val offset = uiPoints[draggedIndex]

                                            if (size.width > 0 && size.height > 0) {
                                                val normalizedX = (offset.x / size.width).coerceIn(0f, 1f)
                                                val normalizedY = (offset.y / size.height).coerceIn(0f, 1f)

                                                println("Normalized: $normalizedX, $normalizedY")
                                                setupViewModel._boundingNormalizedPoints[draggedIndex] =
                                                    Offset(normalizedX, normalizedY)
                                            }

//                                            with(coordinateTransformer) {
//                                                val surfaceCoords = offset.transform()
//                                                val surfaceResolution = it.resolution
//
//                                                val normalized = Offset(
//                                                    (surfaceCoords.x / surfaceResolution.width).coerceIn(0.0f, 1.0f),
//                                                    (surfaceCoords.y / surfaceResolution.height).coerceIn(0.0f, 1.0f)
//                                                )
//
//                                                setupViewModel._boundingNormalizedPoints[draggedIndex] = normalized
//                                            }
                                        }
                                        draggedIndex = -1
                                    },
                                )
                            },
                        surfaceRequest = it,
                        implementationMode = implementationMode,
                        coordinateTransformer = coordinateTransformer,
                        contentScale = ContentScale.Fit
                    )

                    if (uiPoints.size == 2) {
                        val topLeft = Offset(
                            minOf(uiPoints[0].x, uiPoints[1].x),
                            minOf(uiPoints[0].y, uiPoints[1].y)
                        )
                        val bottomRight = Offset(
                            maxOf(uiPoints[0].x, uiPoints[1].x),
                            maxOf(uiPoints[0].y, uiPoints[1].y)
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

                    uiPoints.forEachIndexed { index, center ->
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
            }

            Spacer(modifier = Modifier.height(16.dp))

            Column(
                modifier = Modifier.padding(horizontal = 16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Row(
                    modifier = Modifier.clickable(
                        onClick = {
                            if (uiPoints.size == 2) {
                                setupViewModel.useBoundingBox = !setupViewModel.useBoundingBox
                            }
                        }
                    ),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Checkbox(
                        checked = setupViewModel.useBoundingBox,
                        onCheckedChange = { checked ->
                            if (uiPoints.size == 2) {
                                setupViewModel.useBoundingBox = checked
                            }
                        },
                        enabled = uiPoints.size == 2//hasEnoughPoints
                    )

                    Spacer(modifier = Modifier.width(8.dp))

                    Text(
                        text = "Use Bounding Box",
                        color = if (uiPoints.size == 2) Color.Unspecified else Color.Unspecified.copy(alpha = 0.5f)
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))

                Row {

                    // Back button always visible
                    OutlinedButton(
                        onClick = back,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Back")
                    }

                    Spacer(modifier = Modifier.width(12.dp))

                    if (uiPoints.size == 2 && setupViewModel.useBoundingBox) {
                        Button(
                            onClick = {
                                toNextStep()
                            },
                            modifier = Modifier.weight(1f)
                        ) {
                            Text("Next")
                        }
                    } else {
                        OutlinedButton(
                            onClick = toNextStep,
                            modifier = Modifier.weight(1f)
                        ) {
                            Text("Skip")
                        }
                    }
                }
            }
        }
    }
}