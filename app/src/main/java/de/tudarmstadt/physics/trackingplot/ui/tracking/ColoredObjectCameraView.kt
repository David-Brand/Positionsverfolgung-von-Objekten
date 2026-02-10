package de.tudarmstadt.physics.trackingplot.ui.tracking

import android.Manifest
import android.content.pm.PackageManager
import android.view.MotionEvent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import de.tudarmstadt.physics.trackingplot.ui.plotting.PositionSample
import org.opencv.core.Point
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

@Composable
fun ColoredObjectCameraView(
    onPositionDetected: (PositionSample) -> Unit
) {
    val context = LocalContext.current
    var hasPermission by remember { mutableStateOf(false) }

    val permissionLauncher =
        rememberLauncherForActivityResult(
            contract = ActivityResultContracts.RequestPermission()
        ) { granted ->
            hasPermission = granted
        }

    LaunchedEffect(Unit) {
        if (ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            hasPermission = true
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    if (!hasPermission) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("Camera permission required")
        }
        return
    }


//    var viewSize by remember { mutableStateOf(IntSize.Zero) }
//    var matSize by remember { mutableStateOf(IntSize.Zero) }
//
//    var start by remember { mutableStateOf<Offset?>(null) }
//    var end by remember { mutableStateOf<Offset?>(null) }
    var dragStart by remember { mutableStateOf<Offset?>(null) }
    var dragEnd by remember { mutableStateOf<Offset?>(null) }


    Box(
        modifier = Modifier
            .fillMaxSize()
//            .onSizeChanged { viewSize = it }
            .pointerInput(Unit) {
                detectDragGestures(
                    onDragStart = { offset ->
                        dragStart = offset
                        dragEnd = offset
                    },
                    onDrag = { change, _ ->
                        dragEnd = change.position
                    },
                    onDragEnd = {
//                        val p1 = start
//                        val p2 = end
//                        if (p1 != null && p2 != null && matSize != IntSize.Zero) {
//                            val m1 = screenToMat(p1, viewSize, matSize)
//                            val m2 = screenToMat(p2, viewSize, matSize)
////                            onPointsSelected(m1, m2)
//                        }
                    },
                    onDragCancel = {
                        dragStart = null
                        dragEnd = null
                    }
                )
            }
    ) {

        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { context ->
                OpenCvCameraView(
                    context,
                ).apply {
                    setOnPositionDetected(onPositionDetected)
                    startCamera()
                }
            },
            update = { cameraView ->
                // FORCE CENTER-CROP (removes black bars)
//                cameraView.scaleX = max(
//                    viewSize.width.toFloat() / matSize.width,
//                    viewSize.height.toFloat() / matSize.height
//                )
//                cameraView.scaleY = cameraView.scaleX
            },
            onRelease = { view ->
                view.stopCamera()
            },
        )

        // Overlay selection rectangle
        Canvas(modifier = Modifier.fillMaxSize()) {
            val start = dragStart
            val end = dragEnd

            if (start != null && end != null) {
                drawRect(
                    color = Color.Red,
                    Offset(
                        x = min(start.x, end.x),
                        y = min(start.y, end.y)
                    ),
                    size = Size(
                        width = abs(end.x - start.x),
                        height = abs(end.y - start.y)
                    ),
                    style = Stroke(width = 3f)
//                    topLeft = start!!,
//                    size = Size(
//                        end!!.x - start!!.x,
//                        end!!.y - start!!.y
//                    ),
//                    style = Stroke(3f)
                )
            }
        }
    }
}

fun screenToMat(
    screen: Offset,
    viewSize: IntSize,
    matSize: IntSize
): Point {

    val scale = max(
        viewSize.width.toFloat() / matSize.width,
        viewSize.height.toFloat() / matSize.height
    )

    val scaledWidth = matSize.width * scale
    val scaledHeight = matSize.height * scale

    val cropX = (scaledWidth - viewSize.width) / 2f
    val cropY = (scaledHeight - viewSize.height) / 2f

    val xInMat = (screen.x + cropX) / scale
    val yInMat = (screen.y + cropY) / scale

    return Point(
        xInMat.coerceIn(0f, matSize.width.toFloat()).toDouble(),
        yInMat.coerceIn(0f, matSize.height.toFloat()).toDouble()
    )
}
