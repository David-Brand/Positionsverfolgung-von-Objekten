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
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import de.tudarmstadt.physics.trackingplot.ui.theme.TrackingPlotTheme
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.JavaCameraView
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat

class MainActivity : ComponentActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    companion object {
        private const val TAG = "MainActivity"
    }

    private var mOpenCvCameraView: JavaCameraView? = null
    private var permissionGranted = false

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        permissionGranted = isGranted
        if (isGranted) {
            mOpenCvCameraView?.let { view ->
                view.setCameraPermissionGranted()
                view.enableView()
            }
        } else {
            // Handle permission denied
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContent {
            TrackingPlotTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { paddingValues ->
                    OpenCvCameraScreen(
                        onCameraViewCreated = { cameraView ->
                            mOpenCvCameraView = cameraView
                            cameraView.setCvCameraViewListener(this@MainActivity)

                            // Start camera when we have permission
                            if (hasCameraPermission()) {
                                permissionGranted = true
                                cameraView.setCameraPermissionGranted()
                                cameraView.enableView()
                            } else {
                                requestCameraPermission()
                            }
                        },
                        modifier = Modifier.padding(paddingValues)
                    )
                }
            }
        }

        if (!hasCameraPermission()) {
            requestCameraPermission()
        }
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
    override fun onCameraViewStarted(width: Int, height: Int) {
        // called when camera preview starts
    }

    override fun onCameraViewStopped() {
        // called when camera preview stops
    }

    override fun onCameraFrame(frame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        val mat = frame.rgba()
//        adaptiveThresholdFromJNI(mat.nativeObjAddr)
        highlightRedDot(mat.nativeObjAddr)
        return mat
    }

    private external fun adaptiveThresholdFromJNI(matAddr: Long)
    private external fun highlightRedDot(matAddr: Long)

    override fun onResume() {
        super.onResume()

        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV initialization failed")
            Toast.makeText(this, "OpenCV initialization failed", Toast.LENGTH_LONG).show()
            return
        }

        // Load native library
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
//        mOpenCvCameraView = null
    }
}

@Composable
fun OpenCvCameraScreen(
    onCameraViewCreated: (JavaCameraView) -> Unit = {},
    modifier: Modifier = Modifier
) {
    AndroidView(
        modifier = modifier.fillMaxSize(),
        factory = { ctx ->
            JavaCameraView(ctx, -1).apply {  // 0 = back camera, -1 = any
                // Important: make sure it's visible
                visibility = SurfaceView.VISIBLE

                // You can also set other properties here:
                // setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK)
                // setMaxFrameSize(1280, 720)
                // ...

                onCameraViewCreated(this)
            }
        },
        update = { view ->
            // Called on recomposition - usually not needed for camera
        },
        onRelease = { view ->
            view.disableView()
        }
    )
}