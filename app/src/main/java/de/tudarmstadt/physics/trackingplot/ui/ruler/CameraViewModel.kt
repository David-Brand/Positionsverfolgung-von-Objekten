package de.tudarmstadt.physics.trackingplot.ui.ruler

import android.graphics.PointF
import android.graphics.RectF
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Size
import androidx.lifecycle.ViewModel

class CameraViewModel : ViewModel() {
    // Calibration points in **preview** coordinate system (0..preview.width, 0..preview.height)
    var point1 by mutableStateOf<PointF?>(null)
    var point2 by mutableStateOf<PointF?>(null)

    // Tracking rectangle (preview coordinates)
    var trackingRect by mutableStateOf<RectF?>(null)

    // For live tracking
    var trackedBoundingBox by mutableStateOf<RectF?>(null)

    // You can also keep latest camera frame size here
    var previewSize by mutableStateOf(Size(0f, 0f))
}