package de.tudarmstadt.physics.trackingplot.ui.setup

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.util.Rational
import androidx.camera.core.CameraEffect
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceRequest
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import de.tudarmstadt.physics.trackingplot.DistanceUnit
import de.tudarmstadt.physics.trackingplot.db.ExperimentDatabase
import de.tudarmstadt.physics.trackingplot.tracker2.ColorTrackerConfig
import de.tudarmstadt.physics.trackingplot.tracker2.NativeTracker
import de.tudarmstadt.physics.trackingplot.tracker2.Point2D
import de.tudarmstadt.physics.trackingplot.tracker2.Roi
import de.tudarmstadt.physics.trackingplot.tracker2.Ruler
import de.tudarmstadt.physics.trackingplot.tracker2.TrackingConfig
import de.tudarmstadt.physics.trackingplot.tracker2.TrackingSession
import de.tudarmstadt.physics.trackingplot.tracking.yuvToRgba
import de.tudarmstadt.physics.trackingplot.ui.setup.tracker.NormalizedBox
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.opencv.core.Core
import org.opencv.core.Mat
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class SetupViewModel(
    private val nativeTracker: NativeTracker,
    private val trackingSession: TrackingSession,
    private val db: ExperimentDatabase
): ViewModel() {

    // RULER
    /*private */val _uiPoints = mutableStateListOf<Offset>() //ui offset coordinates
    val uiPoints = _uiPoints as List<Offset>

    /*private */val _normalizedPoints = mutableStateListOf<Offset>() //0.0 - 1.0
    val normalizedPoints = _normalizedPoints as List<Offset>

    var distanceText by mutableStateOf("")
    var selectedUnit by mutableStateOf(DistanceUnit.CENTIMETER)

    var distance by mutableStateOf<Double?>(null)



    // BOUNDING BOX
    val _boundingUiPoints = mutableStateListOf<Offset>()
    val boundingUiPoints = _boundingUiPoints as List<Offset>

    val _boundingNormalizedPoints = mutableStateListOf<Offset>() //0.0 - 1.0
    val boundingNormalizedPoints = _boundingNormalizedPoints as List<Offset>

    var useBoundingBox by mutableStateOf(true)



    // TRACKER

    var imageCapture by mutableStateOf<ImageCapture?>(null)
//    fun onPickColorPressed(
//        imageCapture: ImageCapture,
//        context: Context
//    ) {
//        val executor = ContextCompat.getMainExecutor(context)
//
//        imageCapture.takePicture(
//            executor
//        )
//    }
    var frozenBitmap: Bitmap? by mutableStateOf(null)
    var isPickingColor by mutableStateOf(false)




    val surfaceRequests = MutableStateFlow<SurfaceRequest?>(null)


    fun createUseCaseGroup(
//        cameraInfo: CameraInfo,
        imageCapture: ImageCapture? = null,
        imageAnalysis: ImageAnalysis? = null,
        effect: CameraEffect? = null,
    ): UseCaseGroup {
        val previewUseCase = createPreviewUseCase()

        return UseCaseGroup.Builder().apply {
            setViewPort(ViewPort.Builder(
                Rational(3, 4),
                previewUseCase.targetRotation
            ).build())
            addUseCase(previewUseCase)

            imageCapture?.let { addUseCase(it) }
            imageAnalysis?.let { addUseCase(it) }

//            todo anlysis use case??
            effect?.let { addEffect(it) }
        }.build()
    }

    private fun createPreviewUseCase(): Preview = Preview.Builder().apply {
        setResolutionSelector(
            ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .build()
        )
    }.build().apply {
        setSurfaceProvider { surfaceRequest ->
            surfaceRequests.update { surfaceRequest }
        }
    }


    private val _boxes = MutableStateFlow<List<NormalizedBox>>(emptyList())
    val boxes: StateFlow<List<NormalizedBox>> = _boxes

    fun createImageAnalysis(): ImageAnalysis {
        return ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build().apply {
                setAnalyzer(
                    Executors.newSingleThreadExecutor()
                ) { imageProxy ->
                    try {
                        imageProxy.image?.let {
                            if (it.format == ImageFormat.YUV_420_888 && it.planes.size == 3) {
                                val mat = it.yuvToRgba()
                                when (imageProxy.imageInfo.rotationDegrees) {
                                    90 -> Core.rotate(mat, mat, Core.ROTATE_90_CLOCKWISE)
                                    180 -> Core.rotate(mat, mat, Core.ROTATE_180)
                                    270 -> Core.rotate(mat, mat, Core.ROTATE_90_COUNTERCLOCKWISE)
                                }
                                processFrame(mat)
                            }
                        }
                    } catch (e: IllegalStateException) {
                        e.printStackTrace()
                    } finally {
                        imageProxy.close()
                    }
                }
            }
    }

    private fun Offset.toTopLeftOrigin(): Offset {
        val newX = 1f - this.x // flip X
        val newY = this.y       // Y is already top = 0, bottom = 1 → matches Mat
        return Offset(newX, newY)
    }

    private fun normalizedPointsToRoiBottomLeftOrigin(
        p1: Offset,
        p2: Offset,
        frameWidth: Int,
        frameHeight: Int
    ): Rect {

        // Convert from normalized bottom-left origin to Mat top-left origin
        val x1 = ((1f - p1.x) * frameWidth).roundToInt()
        val y1 = ((1f - p1.y) * frameHeight).roundToInt()

        val x2 = ((1f - p2.x) * frameWidth).roundToInt()
        val y2 = ((1f - p2.y) * frameHeight).roundToInt()

        val left = min(x1, x2).coerceIn(0, frameWidth - 1)
        val top = min(y1, y2).coerceIn(0, frameHeight - 1)
        val right = max(x1, x2).coerceIn(0, frameWidth)
        val bottom = max(y1, y2).coerceIn(0, frameHeight)

        return Rect(left, top, right, bottom)
    }
    private fun calculateRoi(
        p1: Offset,
        p2: Offset,
        frameWidth: Int,
        frameHeight: Int
    ): Rect {
        // Convert normalized to pixel coordinates
        val x1 = (p1.x * frameWidth).roundToInt()
        val y1 = (p1.y * frameHeight).roundToInt()

        val x2 = (p2.x * frameWidth).roundToInt()
        val y2 = (p2.y * frameHeight).roundToInt()

        // Ensure correct ordering regardless of drag direction
        val left = min(x1, x2)
        val right = max(x1, x2)
        val top = min(y1, y2)
        val bottom = max(y1, y2)

        // Clamp to image bounds
        val clampedLeft = left.coerceIn(0, frameWidth - 1)
        val clampedTop = top.coerceIn(0, frameHeight - 1)
        val clampedRight = right.coerceIn(0, frameWidth)
        val clampedBottom = bottom.coerceIn(0, frameHeight)

        return Rect(
            clampedLeft,
            clampedTop,
            clampedRight,
            clampedBottom
        )
    }

    fun normalizedPointsToRect(p1: Offset, p2: Offset, frameWidth: Int, frameHeight: Int): Rect {

        // Convert both points to top-left origin
//        val np1 = Offset(1f - p1.x, p1.y)
//        val np2 = Offset(1f - p2.x, p2.y)
//        val np1 = Offset(p1.x, 1f - p1.y)
//        val np2 = Offset(p2.x, 1f - p2.y)
        val np1 = Offset(p1.x, p1.y)
        val np2 = Offset(p2.x, p2.y)

        // Convert to pixel coordinates
        val x1 = (np1.x * frameWidth).roundToInt()
        val y1 = (np1.y * frameHeight).roundToInt()
        val x2 = (np2.x * frameWidth).roundToInt()
        val y2 = (np2.y * frameHeight).roundToInt()

        // Ensure left < right, top < bottom
        val left = min(x1, x2).coerceIn(0, frameWidth - 1)
        val top = min(y1, y2).coerceIn(0, frameHeight - 1)
        val right = max(x1, x2).coerceIn(0, frameWidth)
        val bottom = max(y1, y2).coerceIn(0, frameHeight)

        return Rect(left, top, right, bottom)
    }

    private fun processFrame(mat: Mat) {

//        val mat = imageProxyToMat(imageProxy) // You said this exists
//        val mat = image.yuvToRgba() // You said this exists

        val frameWidth = mat.width()
        val frameHeight = mat.height()

        val roi = if (useBoundingBox) {
            val points = boundingNormalizedPoints
//            if (points.size == 2) calculateRoi(points[0], points[1], frameWidth, frameHeight)
            if (points.size == 2) normalizedPointsToRect(points[0], points[1], frameWidth, frameHeight)
            else null
        } else null

//        println("ROI : ${roi?.left} ${roi?.top} ${roi?.width()} ${roi?.height()}")
        val results = nativeTracker.updateTrackers(
            mat.nativeObjAddr, roi
        )


        val normalized = results.map { r ->

            val left = r.x.toFloat() / frameWidth
            val top = r.y.toFloat() / frameHeight
            val right = (r.x + r.width).toFloat() / frameWidth
            val bottom = (r.y + r.height).toFloat() / frameHeight
            val centroidX = r.centroidX / frameWidth
            val centroidY = r.centroidY / frameWidth

            NormalizedBox(
                trackerId = r.trackerId,
                left = left,
                top = top,
                right = right,
                bottom = bottom,
                centroidX = centroidX,
                centroidY = centroidY
            )
        }

        _boxes.value = normalized
    }


    override fun onCleared() {
        super.onCleared()
        //TODO reset NativeTracker
    }

    private val _trackers = mutableStateListOf<ColorTrackerConfig>()
    val trackers = _trackers as List<ColorTrackerConfig>
    fun trackerColorSelected(
        index: Int,
        color: Color,
        tolerance: Int
    ) {
        if (_trackers.size == index) {
            _trackers.add(ColorTrackerConfig(
                color = color.toArgb(),
                tolerance = tolerance.toFloat()
            ))
        } else {
            _trackers[index] = ColorTrackerConfig(
                color = color.toArgb(),
                tolerance = tolerance.toFloat()
            )
        }
        nativeTracker.setTracker(
            index = index,
            r = (color.red * 255).toInt(),
            g = (color.green * 255).toInt(),
            b = (color.blue * 255).toInt(),
            tolerance = tolerance
        )
    }

    var samplingRateText by mutableStateOf("10")
    var description by mutableStateOf("")


    fun storeExperimentSetupAndStart() {
        viewModelScope.launch {
            //todo store setup

            val roi = if (useBoundingBox) {
                val points = boundingNormalizedPoints.toList()
//            if (points.size == 2) calculateRoi(points[0], points[1], frameWidth, frameHeight)
                if (points.size == 2) normalizedPointsToRoi(points[0], points[1])
                else null
            } else null

            val rulerPoints = normalizedPoints.toList()
            val realDistance = distanceText.toFloatOrNull()
            val ruler = if (rulerPoints.size == 2 && realDistance != null) {
                //todo
                normalizedPointsToRuler(rulerPoints[0], rulerPoints[1], realDistance, selectedUnit)
            } else null
            val config = TrackingConfig(
                roi,
                ruler,
                _trackers.toList()
            )
            val description = description
            trackingSession.configure(config)
            //on success
//            val experimentId = 123L //todo this is returned by setup store

            val experimentId = db.withTransaction(readOnly = false) {
                val id = addExperiment(config, description)
                id
            }

            eventsChannel.send(UiEvent.ToLiveExperiment(experimentId))
        }
    }

    fun normalizedPointsToRoi(p1: Offset, p2: Offset): Roi {
        val np1 = Offset(p1.x, p1.y)
        val np2 = Offset(p2.x, p2.y)

        // Ensure left < right, top < bottom
        val left = min(np1.x, np2.x).coerceIn(0.0f, 1.0f)
        val top = min(np1.y, np2.y).coerceIn(0.0f, 1.0f)
        val right = max(np1.x, np2.x).coerceIn(0.0f, 1.0f)
        val bottom = max(np1.y, np2.y).coerceIn(0.0f, 1.0f)

        return Roi(
            Point2D(left, top),
            Point2D(right, bottom),
        )
    }

    fun normalizedPointsToRuler(p1: Offset, p2: Offset, realDistance: Float, unit: DistanceUnit): Ruler {
        val np1 = Offset(p1.x, p1.y)
        val np2 = Offset(p2.x, p2.y)

        // Ensure left < right, top < bottom
        val left = min(np1.x, np2.x).coerceIn(0.0f, 1.0f)
        val top = min(np1.y, np2.y).coerceIn(0.0f, 1.0f)
        val right = max(np1.x, np2.x).coerceIn(0.0f, 1.0f)
        val bottom = max(np1.y, np2.y).coerceIn(0.0f, 1.0f)

        return Ruler(
            Point2D(left, top),
            Point2D(right, bottom),
            realDistance,
            unit
        )
    }


    private val eventsChannel = Channel<UiEvent>()
    val eventsChannelFlow = eventsChannel.receiveAsFlow()

    sealed interface UiEvent {
        data class ToLiveExperiment(val experimentId: Long): UiEvent
    }
}