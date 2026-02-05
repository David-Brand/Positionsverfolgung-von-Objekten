package de.tudarmstadt.physics.trackingplot.ui.tracking

import android.content.Context
import android.view.SurfaceView
import de.tudarmstadt.physics.trackingplot.tracking.NativeTracker
import de.tudarmstadt.physics.trackingplot.ui.plotting.PositionSample
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.JavaCameraView
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

class OpenCvCameraView(
    context: Context
) : JavaCameraView(context, CAMERA_ID_BACK),
    CameraBridgeViewBase.CvCameraViewListener2 {

    private var positionListener: ((PositionSample) -> Unit)? = null

    fun setOnPositionDetected(listener: (PositionSample) -> Unit) {
        positionListener = listener
    }

    init {
        visibility = SurfaceView.VISIBLE
        setCvCameraViewListener(this)
//        enableView()
    }

    fun startCamera() {
        setCameraPermissionGranted()
        enableView()
    }

    fun stopCamera() {
        disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {}
    override fun onCameraViewStopped() {}

    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat {
        val rgba = inputFrame.rgba()

        val result = NativeTracker.processFrame(
            rgba.nativeObjAddr,
            rgba.width(),
            rgba.height()
        )

        result?.let {
            positionListener?.invoke(
                PositionSample(
                    timeMs = System.currentTimeMillis(),
                    x = it[0],
                    y = it[1]
                )
            )
        }

        return rgba
//        val rgba = inputFrame.rgba()
//        val hsv = Mat()
//
//        Imgproc.cvtColor(rgba, hsv, Imgproc.COLOR_RGB2HSV)
//
//        val mask1 = Mat()
//        val mask2 = Mat()
//
//        Core.inRange(
//            hsv,
//            Scalar(0.0, 120.0, 70.0),
//            Scalar(10.0, 255.0, 255.0),
//            mask1
//        )
//
//        Core.inRange(
//            hsv,
//            Scalar(170.0, 120.0, 70.0),
//            Scalar(180.0, 255.0, 255.0),
//            mask2
//        )
//
//        val mask = Mat()
//        Core.add(mask1, mask2, mask)
//
//        val contours = ArrayList<MatOfPoint>()
//        Imgproc.findContours(
//            mask,
//            contours,
//            Mat(),
//            Imgproc.RETR_EXTERNAL,
//            Imgproc.CHAIN_APPROX_SIMPLE
//        )
//
//        val largest = contours.maxByOrNull { Imgproc.contourArea(it) }
//
//        largest?.let {
//            val moments = Imgproc.moments(it)
//            if (moments.m00 != 0.0) {
//                val cx = (moments.m10 / moments.m00).toFloat()
//                val cy = (moments.m01 / moments.m00).toFloat()
//
//                val normX = cx / rgba.width()
//                val normY = cy / rgba.height()
//
//                listener?.invoke(
//                    PositionSample(
//                        timeMs = System.currentTimeMillis(),
//                        x = normX,
//                        y = normY
//                    )
//                )
//
//                Imgproc.circle(
//                    rgba,
//                    Point(cx, cy),
//                    10,
//                    Scalar(0.0, 255.0, 0.0),
//                    2
//                )
//            }
//        }
//
//        return rgba
    }
}
