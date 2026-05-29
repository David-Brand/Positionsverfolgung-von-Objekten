package de.tudarmstadt.physics.trackingplot.ui.live_experiment

import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.util.Rational
import androidx.camera.core.Camera
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceRequest
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.mutableStateSetOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import de.tudarmstadt.physics.trackingplot.db.ExperimentDatabase
import de.tudarmstadt.physics.trackingplot.tracker2.NativeTracker
import de.tudarmstadt.physics.trackingplot.tracker2.TrackingConfig
import de.tudarmstadt.physics.trackingplot.tracker2.TrackingSession
import de.tudarmstadt.physics.trackingplot.tracking.yuvToRgba
import de.tudarmstadt.physics.trackingplot.ui.setup.tracker.NormalizedBox
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.opencv.core.Core
import org.opencv.core.Mat
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class LiveExperimentViewModel(
    val experimentId: Long,
    private val trackingSession: TrackingSession,
    private val nativeTracker: NativeTracker,
    private val db: ExperimentDatabase
): ViewModel() {

    val surfaceRequests = MutableStateFlow<SurfaceRequest?>(null)

    var camera: Camera? = null

    private val _boxes = MutableStateFlow<List<NormalizedBox>>(emptyList())
    val boxes: StateFlow<List<NormalizedBox>> = _boxes

    var boundingBoxPoints by mutableStateOf<Pair<Offset, Offset>?>(null)
        private set

    init {
        val config = trackingSession.getConfig()
        config.roi?.let {
            boundingBoxPoints =
                Offset(it.p1.x, it.p1.y) to Offset(it.p2.x, it.p2.y)
        }
//        boundingBoxPoints =
//            Offset(0.2f, 0.2f) to Offset(0.8f, 0.8f)
    }

    override fun onCleared() {
        super.onCleared()
        nativeTracker.reset()
        trackingSession.reset()
    }


    private var videoCapture: VideoCapture<Recorder>? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var recording: Recording? = null

    var isRecording by mutableStateOf(trackingSession.isActive())
        private set

    fun createUseCaseGroup(
    ): UseCaseGroup {
        val previewUseCase = createPreviewUseCase()

        return UseCaseGroup.Builder().apply {
            setViewPort(ViewPort.Builder(
                Rational(3, 4),
                previewUseCase.targetRotation
            ).build())
            addUseCase(previewUseCase)

            val imageAnalysis = createImageAnalysis()
            this@LiveExperimentViewModel.imageAnalysis = imageAnalysis
            addUseCase(imageAnalysis)

            val recorder = Recorder.Builder().apply {
                //TODO
            }.build()
            val videoCapture = VideoCapture.withOutput(recorder)
            this@LiveExperimentViewModel.videoCapture = videoCapture
            addUseCase(videoCapture)
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


    fun createImageAnalysis(): ImageAnalysis {
        return ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build().apply {
                setAnalyzer(
                    Executors.newSingleThreadExecutor()
                ) { imageProxy ->
                    try {
                        if (trackingSession.isActive()) {
                            imageProxy.image?.let {
                                if (it.format == ImageFormat.YUV_420_888 && it.planes.size == 3) {
                                    val mat = it.yuvToRgba()
                                    when (imageProxy.imageInfo.rotationDegrees) {
                                        90 -> Core.rotate(mat, mat, Core.ROTATE_90_CLOCKWISE)
                                        180 -> Core.rotate(mat, mat, Core.ROTATE_180)
                                        270 -> Core.rotate(mat, mat, Core.ROTATE_90_COUNTERCLOCKWISE)
                                    }
                                    val result = processFrame(mat)

                                    val timestamp = System.currentTimeMillis()

                                    addTrackingResult(timestamp, result)
                                }
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


    fun toggleRecording(context: Context) {
        if (trackingSession.isActive()) {
            recording?.stop()
            trackingSession.stop()
            isRecording = false
        } else {
            trackingSession.start()
            isRecording = true
            baseTime = System.currentTimeMillis()
//            startRecording(context)
        }
    }

    private fun startRecording(context: Context) {
        return.also {
            trackingSession.start()
            isRecording = true
        }
        val file = File(
            context.filesDir,
            "video_${System.currentTimeMillis()}.mp4"
        )

        val outputOptions = FileOutputOptions.Builder(file).build()

        recording = videoCapture?.output
            ?.prepareRecording(context, outputOptions)
            ?.start(ContextCompat.getMainExecutor(context)) { event ->
                if (event is VideoRecordEvent.Start) {
                    isRecording = true
                }
            }
    }

//    fun createVideoCapture(): VideoCapture {
//        TODO()
//    }

    fun normalizedPointsToRect(p1: Offset, p2: Offset, frameWidth: Int, frameHeight: Int): Rect {
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

    private fun processFrame(mat: Mat): List<NormalizedBox> {
        val frameWidth = mat.width()
        val frameHeight = mat.height()

        val roi = boundingBoxPoints?.let { (point0, point1) ->
            normalizedPointsToRect(point0, point1, frameWidth, frameHeight)
        }
        val results = nativeTracker.updateTrackers(
            mat.nativeObjAddr, roi
        )


        val normalized = results.mapNotNull { r ->

            if (!r.found) return@mapNotNull null

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
        return normalized
    }


    /**
     * TODO TEST
     */
    private val _trackingConfig = mutableStateOf<TrackingConfig?>(null)
    val trackingConfig by _trackingConfig
    init {
        //todo load from db
        _trackingConfig.value = trackingSession.getConfig()
    }

    private val _disabledLines = mutableStateSetOf<String>()
    val disabledLines = _disabledLines as Set<String>
    var baseTime = System.currentTimeMillis()
    private val _mostRecentTime = mutableStateOf(baseTime)
    val mostRecentTime by _mostRecentTime

    private val _timeWindowSlider = mutableStateOf(1.0f)
    val timeWindowSlider by _timeWindowSlider
    fun onTimeWindowSliderValueChange(value: Float) {
        _timeWindowSlider.value = value//.coerceIn(0.0f .. 1.0f)
    }
    private val _offsetSlider = mutableStateOf(1.0f)
    val offsetSlider by _offsetSlider
    fun onOffsetSliderValueChange(value: Float) {
        _offsetSlider.value = value//.coerceIn(0.0f .. 1.0f)
    }

    val timeWindowMs by derivedStateOf {
        logarithmicTimeWindowMs(_timeWindowSlider.value)
    }
    val offsetMs by derivedStateOf {
        (_offsetSlider.value * (_mostRecentTime.value - baseTime).coerceAtLeast(0L)).toLong()
    }
    val data = LineData()

    fun addTrackingResult(timestamp: Long, results: List<NormalizedBox>) {
        val nullTimestamp = timestamp - baseTime
        val time = (timestamp - baseTime).coerceAtLeast(0L).toFloat()
        for (result in results) {
            val color = when (result.trackerId) {
                0 -> Color.Red
                1 -> Color.Green
                2 -> Color.Blue
                else -> Color.Gray
            }.toArgb()
//            val color = Color.Red.toArgb()
            val labelX = "${result.trackerId}_X"
            val labelY = "${result.trackerId}_Y"
            val dataSetX = data.getDataSetByLabel(labelX, true) as? LineDataSet
                ?: LineDataSet(mutableListOf(), labelX).apply {
                    setDrawCircles(false)
                    this.color = color
                    data.addDataSet(this)
                }
            val dataSetY = data.getDataSetByLabel(labelY, true) as? LineDataSet
                ?: LineDataSet(mutableListOf(), labelY).apply {
                    setDrawCircles(false)
                    this.color = color
                    data.addDataSet(this)
                }

//            val x = (result.left + result.right) / 2
//            val y = (result.top + result.bottom) / 2
            val x = result.centroidX
            val y = result.centroidY
            dataSetX.addEntry(Entry(time, x.toFloat()))
            dataSetY.addEntry(Entry(time, y.toFloat()))

            dataSetX.notifyDataSetChanged()
            dataSetY.notifyDataSetChanged()
        }
        viewModelScope.launch {
            db.withTransaction(readOnly = false) {
                addMeasurements(
                    experimentId,
                    nullTimestamp,
                    results
                )
            }
        }
        data.notifyDataChanged()
    }

    fun toggleLabel(label: String, visible: Boolean) {
        val dataSet = data.getDataSetByLabel(label, true) as? LineDataSet
            ?: LineDataSet(mutableListOf(), label).apply {
                setDrawCircles(false)
                this.color = color
                data.addDataSet(this)
            }

        dataSet.isVisible = visible

        dataSet.notifyDataSetChanged()

        data.notifyDataChanged()
    }
    fun toggleLine(label: String) {
        val isCurrentlyVisible = !_disabledLines.contains(label)

        val dataSet = data.getDataSetByLabel(label, true) as? LineDataSet
            ?: return
//            ?: LineDataSet(mutableListOf(), label).apply {
//                setDrawCircles(false)
//                this.color = color
//                data.addDataSet(this)
//            }

        if (isCurrentlyVisible) {
            dataSet.isVisible = false
            _disabledLines.add(label)
        } else {
            dataSet.isVisible = true
            _disabledLines.remove(label)
        }

        dataSet.notifyDataSetChanged()

        data.notifyDataChanged()
    }

    fun logarithmicTimeWindowMs(fraction: Float): Long {
        val minMs = 1_000L            // 1 second
        val maxMs = 6 * 60 * 60_000L  // 6 hours

        val logMin = ln(minMs.toDouble())
        val logMax = ln(maxMs.toDouble())

        return exp(logMin + fraction * (logMax - logMin)).toLong()
    }
}