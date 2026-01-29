package de.tudarmstadt.physics.trackingplot

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import de.tudarmstadt.physics.trackingplot.ui.theme.TrackingPlotTheme
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.JavaCameraView
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import kotlin.math.roundToInt
import androidx.compose.ui.unit.dp


class MainActivity : ComponentActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    @Volatile private var pendingHueInitIndex: Int = -1
    @Volatile private var pendingHueInitPoint: IntArray = intArrayOf(0, 0) // [x,y]

    companion object {
        private const val TAG = "MainActivity"
    }

    private var mOpenCvCameraView: JavaCameraView? = null
    private var permissionGranted = false

    // Native state holder (C++ will store a pointer here)
    @Suppress("unused")
    private var nativeTrackerPtr: Long = 0L

    // Shared state used by camera thread (avoid Compose state directly here)
    @Volatile private var roiRectInt: IntArray? = null        // [x,y,w,h] in view pixels
    @Volatile private var boxesInt: IntArray = IntArray(0)    // N*4 [x,y,w,h...]
    @Volatile private var reinitTracking: Boolean = false     // set true when user adds/changes objects

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        permissionGranted = isGranted
        if (isGranted) {
            mOpenCvCameraView?.let { view ->
                view.setCameraPermissionGranted()
                view.enableView()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContent {
            TrackingPlotTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { paddingValues ->
                    CameraWithTrackingOverlay(
                        modifier = Modifier.padding(paddingValues),
                        onCameraViewCreated = { cameraView ->
                            mOpenCvCameraView = cameraView
                            cameraView.setCvCameraViewListener(this@MainActivity)

                            if (hasCameraPermission()) {
                                permissionGranted = true
                                cameraView.setCameraPermissionGranted()
                                cameraView.enableView()
                            } else {
                                requestCameraPermission()
                            }
                        },
                        onRoiChanged = { roi ->
                            roiRectInt = roi
                            // changing ROI usually means reinit (so centers are ROI-relative)
                            reinitTracking = true
                        },
                        onBoxesChanged = { newBoxes ->
                            boxesInt = newBoxes
                            reinitTracking = true
                        },
                        onHueInitRequested = { idx, x, y ->
                            pendingHueInitIndex = idx
                            pendingHueInitPoint = intArrayOf(x, y)
                        }
                    )
                }
            }
        }

        if (!hasCameraPermission()) requestCameraPermission()
    }

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermission() {
        cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
    }

    // ───────────────────────────────────────────────
    //     CvCameraViewListener2 implementation
    // ───────────────────────────────────────────────
    override fun onCameraViewStarted(width: Int, height: Int) {}
    override fun onCameraViewStopped() {}

    override fun onCameraFrame(frame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        val mat = frame.rgba()

        val roi = roiRectInt
        val boxes = boxesInt

        if (roi != null && boxes.isNotEmpty()) {
            val doReinit = reinitTracking
            if (doReinit) reinitTracking = false

            val idx = pendingHueInitIndex
            val pt = pendingHueInitPoint
            // clear immediately so we do it only once
            pendingHueInitIndex = -1

            nativeTrack(mat.nativeObjAddr, roi, boxes, doReinit, idx, pt)
            boxesInt = boxes
        }

        return mat
    }


    private external fun nativeTrack(
        matAddr: Long,
        roi: IntArray,
        boxesInOut: IntArray,
        reinit: Boolean,
        hueInitIndex: Int,       // -1 if none
        hueInitPoint: IntArray   // [x,y] in view/frame coords (ignored if hueInitIndex==-1)
    ): Boolean


    private external fun nativeRelease()

    override fun onResume() {
        super.onResume()

        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV initialization failed")
            Toast.makeText(this, "OpenCV initialization failed", Toast.LENGTH_LONG).show()
            return
        }

        try {
            System.loadLibrary("native-lib")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Failed to load native library: ${e.message}")
            Toast.makeText(this, "Failed to load native library", Toast.LENGTH_LONG).show()
            return
        }

        mOpenCvCameraView?.enableView()
    }

    override fun onPause() {
        super.onPause()
        mOpenCvCameraView?.disableView()
    }

    override fun onDestroy() {
        super.onDestroy()
        mOpenCvCameraView?.disableView()
        nativeRelease()
    }
}

@Composable
private fun CameraWithTrackingOverlay(
    modifier: Modifier = Modifier,
    onCameraViewCreated: (JavaCameraView) -> Unit,
    onRoiChanged: (IntArray) -> Unit,
    onBoxesChanged: (IntArray) -> Unit,
    onHueInitRequested: (index: Int, x: Int, y: Int) -> Unit
) {
    // ROI corners (4 taps)
    var roiPoints by remember { mutableStateOf(listOf<Offset>()) }
    var roiRect by remember { mutableStateOf<IntArray?>(null) }

    // Tracked boxes in UI state as a list of IntArray(4)
    var uiBoxes by remember { mutableStateOf(listOf<IntArray>()) }

    // Object adding UI
    var addingObject by remember { mutableStateOf(false) }
    var selectorCenter by remember { mutableStateOf(Offset(200f, 200f)) }

    val density = LocalDensity.current
    val defaultBoxSizePx = with(density) { 40.dp.toPx() } // initial box size around selector
    val selectorRadiusPx = with(density) { 10.dp.toPx() }

    Box(modifier = modifier.fillMaxSize()) {

        // Camera preview
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                JavaCameraView(ctx, CameraBridgeViewBase.CAMERA_ID_BACK).apply {
                    visibility = SurfaceView.VISIBLE
                    setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK)
                    onCameraViewCreated(this)
                }
            },
            onRelease = { it.disableView() }
        )

        // Overlay gestures for ROI selection (tap 4 corners)
        Box(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(roiPoints) {
                    detectTapGestures { pos ->
                        if (roiRect == null) {
                            val newPts = (roiPoints + pos).take(4)
                            roiPoints = newPts
                            if (newPts.size == 4) {
                                val minX = newPts.minOf { it.x }
                                val minY = newPts.minOf { it.y }
                                val maxX = newPts.maxOf { it.x }
                                val maxY = newPts.maxOf { it.y }
                                val rect = intArrayOf(
                                    minX.roundToInt(),
                                    minY.roundToInt(),
                                    (maxX - minX).roundToInt(),
                                    (maxY - minY).roundToInt()
                                )
                                roiRect = rect
                                onRoiChanged(rect)
                            }
                        }
                    }
                }
        )

        // Draw ROI + boxes (UI overlay). Native also draws, but this helps UX.
        Canvas(modifier = Modifier.fillMaxSize()) {
            // ROI points
            for (p in roiPoints) {
                drawCircle(
                    color = androidx.compose.ui.graphics.Color.Yellow,
                    radius = 6f,
                    center = p
                )
            }
            // ROI rect
            roiRect?.let { r ->
                drawRect(
                    color = androidx.compose.ui.graphics.Color.Red,
                    topLeft = Offset(r[0].toFloat(), r[1].toFloat()),
                    size = androidx.compose.ui.geometry.Size(r[2].toFloat(), r[3].toFloat()),
                    style = Stroke(width = 3f)
                )
            }
            // Tracked boxes
            uiBoxes.forEach { b ->
                drawRect(
                    color = androidx.compose.ui.graphics.Color.Red,
                    topLeft = Offset(b[0].toFloat(), b[1].toFloat()),
                    size = androidx.compose.ui.geometry.Size(b[2].toFloat(), b[3].toFloat()),
                    style = Stroke(width = 3f)
                )
            }
        }

        // + button top-left
        IconButton(
            onClick = {
                if (roiRect != null) {
                    addingObject = true
                    // start selector roughly center of ROI if available
                    roiRect?.let { r ->
                        selectorCenter = Offset(
                            (r[0] + r[2] / 2).toFloat(),
                            (r[1] + r[3] / 2).toFloat()
                        )
                    }
                }
            },
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(12.dp)
        ) {
            Icon(Icons.Filled.Add, contentDescription = "Add object")
        }

        // If ROI not set, show a small hint
        if (roiRect == null) {
            Surface(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 12.dp),
                tonalElevation = 3.dp
            ) {
                Text(
                    text = "Tap 4 corners to set ROI",
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)
                )
            }
        }

        // Object selector: draggable circle + confirm button next to it
        if (addingObject && roiRect != null) {
            // Draggable circle
            Box(
                modifier = Modifier
                    .offset(
                        x = with(density) { (selectorCenter.x - selectorRadiusPx).toDp() },
                        y = with(density) { (selectorCenter.y - selectorRadiusPx).toDp() }
                    )
                    .size(with(density) { (2 * selectorRadiusPx).toDp() })
                    .pointerInput(Unit) {
                        detectDragGestures { change, drag ->
                            change.consume()
                            selectorCenter += drag
                        }
                    },
                contentAlignment = Alignment.Center
            ) {
                Surface(
                    shape = CircleShape,
                    color = androidx.compose.ui.graphics.Color.Transparent,
                    tonalElevation = 0.dp,
                    modifier = Modifier.fillMaxSize()
                ) {
                    // border drawn via Canvas
                    Canvas(modifier = Modifier.fillMaxSize()) {
                        drawCircle(
                            color = androidx.compose.ui.graphics.Color.White,
                            radius = size.minDimension / 2,
                            style = Stroke(width = 3f)
                        )
                    }
                }
            }

            // Confirm button next to circle
            Button(
                onClick = {
                    val r = roiRect!!

                    // Create initial box around selector center
                    val half = (defaultBoxSizePx / 2f)
                    val x = (selectorCenter.x - half).roundToInt()
                    val y = (selectorCenter.y - half).roundToInt()
                    val w = defaultBoxSizePx.roundToInt()
                    val h = defaultBoxSizePx.roundToInt()

                    // Clamp to ROI
                    val cx = x.coerceIn(r[0], r[0] + r[2] - 1)
                    val cy = y.coerceIn(r[1], r[1] + r[3] - 1)
                    val cw = w.coerceAtMost((r[0] + r[2]) - cx).coerceAtLeast(2)
                    val ch = h.coerceAtMost((r[1] + r[3]) - cy).coerceAtLeast(2)

                    val newBox = intArrayOf(cx, cy, cw, ch)
                    val newUiBoxes = uiBoxes + newBox
                    uiBoxes = newUiBoxes

                    val newIndex = newUiBoxes.lastIndex
                    onHueInitRequested(
                        newIndex,
                        selectorCenter.x.roundToInt(),
                        selectorCenter.y.roundToInt()
                    )


                    // Flatten to N*4 array for native
                    val flat = IntArray(newUiBoxes.size * 4)
                    newUiBoxes.forEachIndexed { i, b ->
                        flat[i * 4 + 0] = b[0]
                        flat[i * 4 + 1] = b[1]
                        flat[i * 4 + 2] = b[2]
                        flat[i * 4 + 3] = b[3]
                    }
                    onBoxesChanged(flat)

                    addingObject = false
                },
                modifier = Modifier
                    .offset(
                        x = with(density) { (selectorCenter.x + selectorRadiusPx + 10f).toDp() },
                        y = with(density) { (selectorCenter.y - 18f).toDp() }
                    )
                    .height(36.dp)
            ) {
                Text("OK")
            }
        }

        // Optional: small reset ROI button bottom-left
        TextButton(
            onClick = {
                roiPoints = emptyList()
                roiRect = null
                uiBoxes = emptyList()
                onBoxesChanged(IntArray(0))
            },
            modifier = Modifier
                .align(Alignment.BottomStart)
                .padding(12.dp)
        ) {
            Text("Reset")
        }
    }
}
