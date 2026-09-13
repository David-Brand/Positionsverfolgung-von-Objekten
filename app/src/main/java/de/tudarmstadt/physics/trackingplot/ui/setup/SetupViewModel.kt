package de.tudarmstadt.physics.trackingplot.ui.setup

import android.graphics.Bitmap
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import de.tudarmstadt.physics.trackingplot.camera.CameraPipeline
import de.tudarmstadt.physics.trackingplot.db.DatabaseException
import de.tudarmstadt.physics.trackingplot.db.ExperimentDatabase
import de.tudarmstadt.physics.trackingplot.geometry.RulerScale
import de.tudarmstadt.physics.trackingplot.model.ColorTrackerConfig
import de.tudarmstadt.physics.trackingplot.model.DistanceUnit
import de.tudarmstadt.physics.trackingplot.model.HsvColor
import de.tudarmstadt.physics.trackingplot.model.HsvTolerance
import de.tudarmstadt.physics.trackingplot.model.Point2D
import de.tudarmstadt.physics.trackingplot.model.Roi
import de.tudarmstadt.physics.trackingplot.model.Ruler
import de.tudarmstadt.physics.trackingplot.model.TrackedObject
import de.tudarmstadt.physics.trackingplot.model.TrackingConfig
import de.tudarmstadt.physics.trackingplot.tracker.NativeTracker
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

/**
 * State of the three setup steps (ruler, region of interest, trackers). Shared by the setup
 * screens; scoped to the setup navigation graph.
 *
 * All points are canonical frame coordinates (see FrameGeometry); the screens convert to and
 * from screen positions with a FrameMapper.
 */
class SetupViewModel(
    private val db: ExperimentDatabase
) : ViewModel() {

    data class TrackerSlot(
        /** Picked colour (ARGB) or null while nothing has been picked yet. */
        val color: Int?,
        val tolerance: Int = ColorTrackerConfig.DEFAULT_TOLERANCE
    ) {
        val hsv: HsvColor? get() = color?.let { HsvColor.fromArgb(it) }
    }

    enum class RulerProblem { NEED_POINTS, NEED_DISTANCE, INVALID_DISTANCE, DISTANCE_NOT_POSITIVE, POINTS_TOO_CLOSE }

    sealed interface RulerState {
        /** No points and no distance entered: the experiment runs without real-world units. */
        data object NotSet : RulerState
        data class Incomplete(val problem: RulerProblem) : RulerState
        data class Valid(val ruler: Ruler) : RulerState
    }

    enum class StartProblem { NO_TRACKER, TRACKER_WITHOUT_COLOR, INVALID_SAMPLING_RATE }

    sealed interface UiEvent {
        data class ToLiveExperiment(val experimentId: Long) : UiEvent
    }

    // ---------------------------------------------------------------- camera

    private val tracker = NativeTracker()

    @Volatile
    private var trackerPreviewEnabled = false

    val pipeline = CameraPipeline(tracker) { trackerPreviewEnabled }

    /** Boxes of the currently tracked objects (only while the tracker preview is enabled). */
    val trackedObjects: StateFlow<List<TrackedObject>> = pipeline.frames
        .map { it.objects }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(1_000), emptyList())

    /** The tracker setup screen enables live tracking while it is visible. */
    fun setTrackerPreviewEnabled(enabled: Boolean) {
        trackerPreviewEnabled = enabled
    }

    val zoomRatio: StateFlow<Float> = pipeline.zoomRatio
    val zoomRange: StateFlow<ClosedFloatingPointRange<Float>> = pipeline.zoomRange

    fun setZoomRatio(ratio: Float) = pipeline.setZoomRatio(ratio)

    fun zoomBy(factor: Float) = pipeline.setZoomRatio(pipeline.zoomRatio.value * factor)

    // ---------------------------------------------------------------- ruler

    private val _rulerPoints = mutableStateListOf<Point2D>()
    val rulerPoints: List<Point2D> get() = _rulerPoints

    var distanceText by mutableStateOf("")
    var distanceUnit by mutableStateOf(DistanceUnit.CENTIMETER)

    /** Distance parsed from [distanceText]; accepts a decimal comma. Null when not a number. */
    val parsedDistance: Float?
        get() = distanceText.trim().replace(',', '.').toFloatOrNull()?.takeIf { it.isFinite() }

    val rulerState: RulerState
        get() {
            val hasInput = _rulerPoints.isNotEmpty() || distanceText.isNotBlank()
            if (!hasInput) return RulerState.NotSet
            if (_rulerPoints.size < 2) return RulerState.Incomplete(RulerProblem.NEED_POINTS)
            if (distanceText.isBlank()) return RulerState.Incomplete(RulerProblem.NEED_DISTANCE)
            val distance = parsedDistance ?: return RulerState.Incomplete(RulerProblem.INVALID_DISTANCE)
            if (distance <= 0f) return RulerState.Incomplete(RulerProblem.DISTANCE_NOT_POSITIVE)
            val ruler = Ruler(_rulerPoints[0], _rulerPoints[1], distance, distanceUnit)
            if (!RulerScale.isValid(ruler)) return RulerState.Incomplete(RulerProblem.POINTS_TOO_CLOSE)
            return RulerState.Valid(ruler)
        }

    fun addRulerPoint(point: Point2D) {
        if (_rulerPoints.size < 2) _rulerPoints.add(point)
    }

    fun moveRulerPoint(index: Int, point: Point2D) {
        if (index in _rulerPoints.indices) _rulerPoints[index] = point
    }

    fun clearRulerPoints() = _rulerPoints.clear()

    // ---------------------------------------------------------------- region of interest

    private val _roiPoints = mutableStateListOf<Point2D>()
    val roiPoints: List<Point2D> get() = _roiPoints

    var useRoi by mutableStateOf(true)

    /** The region that will be used, or null (whole frame) if disabled, incomplete or too small. */
    val roi: Roi?
        get() {
            if (!useRoi || _roiPoints.size < 2) return null
            val roi = Roi(_roiPoints[0], _roiPoints[1])
            val wideEnough = roi.right - roi.left >= MIN_ROI_SIZE
            val tallEnough = roi.bottom - roi.top >= MIN_ROI_SIZE
            return if (wideEnough && tallEnough) roi else null
        }

    /** True when two points are set but the region between them is too small to be useful. */
    val roiTooSmall: Boolean
        get() = _roiPoints.size == 2 && Roi(_roiPoints[0], _roiPoints[1]).let {
            it.right - it.left < MIN_ROI_SIZE || it.bottom - it.top < MIN_ROI_SIZE
        }

    fun addRoiPoint(point: Point2D) {
        if (_roiPoints.size < 2) _roiPoints.add(point)
    }

    fun moveRoiPoint(index: Int, point: Point2D) {
        if (index in _roiPoints.indices) _roiPoints[index] = point
    }

    fun clearRoiPoints() = _roiPoints.clear()

    // ---------------------------------------------------------------- trackers

    private val _trackerSlots = mutableStateListOf(TrackerSlot(color = null))
    val trackerSlots: List<TrackerSlot> get() = _trackerSlots

    val canAddTracker: Boolean get() = _trackerSlots.size < ColorTrackerConfig.MAX_TRACKERS

    fun addTrackerSlot() {
        if (canAddTracker) _trackerSlots.add(TrackerSlot(color = null))
    }

    fun removeTrackerSlot(index: Int) {
        if (index !in _trackerSlots.indices || _trackerSlots.size == 1) return
        _trackerSlots.removeAt(index)
        if (pickingSlot == index) cancelColorPick()
        syncNativeTrackers()
    }

    fun setTrackerTolerance(index: Int, tolerance: Int) {
        if (index !in _trackerSlots.indices) return
        val clamped = tolerance.coerceIn(ColorTrackerConfig.MIN_TOLERANCE, ColorTrackerConfig.MAX_TOLERANCE)
        _trackerSlots[index] = _trackerSlots[index].copy(tolerance = clamped)
        syncNativeTrackers()
    }

    /** Index of the slot a colour is being picked for; the picker UI is shown while non-null. */
    var pickingSlot by mutableStateOf<Int?>(null)
        private set

    /** Frozen camera frame (canonical orientation) to pick the colour from. */
    var snapshot by mutableStateOf<Bitmap?>(null)
        private set

    var isCapturingSnapshot by mutableStateOf(false)
        private set

    var snapshotError by mutableStateOf<Throwable?>(null)
        private set

    fun startColorPick(slotIndex: Int) {
        if (slotIndex !in _trackerSlots.indices || isCapturingSnapshot) return
        viewModelScope.launch {
            isCapturingSnapshot = true
            snapshotError = null
            try {
                val bitmap = pipeline.captureSnapshot()
                snapshot = bitmap
                pickingSlot = slotIndex
            } catch (e: Exception) {
                snapshotError = e
            } finally {
                isCapturingSnapshot = false
            }
        }
    }

    fun clearSnapshotError() {
        snapshotError = null
    }

    fun confirmColorPick(argb: Int) {
        val index = pickingSlot ?: return
        if (index in _trackerSlots.indices) {
            _trackerSlots[index] = _trackerSlots[index].copy(color = argb)
        }
        cancelColorPick()
        syncNativeTrackers()
    }

    fun cancelColorPick() {
        pickingSlot = null
        snapshot = null
    }

    private fun syncNativeTrackers() {
        tracker.clear()
        _trackerSlots.forEachIndexed { index, slot ->
            val hsv = slot.hsv ?: return@forEachIndexed
            tracker.setTracker(index, hsv, HsvTolerance.fromUiTolerance(slot.tolerance))
        }
    }

    // ---------------------------------------------------------------- start

    var description by mutableStateOf("")
    var samplingRateText by mutableStateOf(TrackingConfig.DEFAULT_SAMPLING_RATE.toString())

    /** Sampling rate parsed from [samplingRateText], or null when out of range. */
    val samplingRate: Int?
        get() = samplingRateText.trim().toIntOrNull()
            ?.takeIf { it in TrackingConfig.MIN_SAMPLING_RATE..TrackingConfig.MAX_SAMPLING_RATE }

    val startProblem: StartProblem?
        get() = when {
            _trackerSlots.none { it.color != null } -> StartProblem.NO_TRACKER
            _trackerSlots.any { it.color == null } -> StartProblem.TRACKER_WITHOUT_COLOR
            samplingRate == null -> StartProblem.INVALID_SAMPLING_RATE
            else -> null
        }

    var isStarting by mutableStateOf(false)
        private set

    var startError by mutableStateOf<Throwable?>(null)
        private set

    fun clearStartError() {
        startError = null
    }

    private val eventsChannel = Channel<UiEvent>(Channel.BUFFERED)
    val events = eventsChannel.receiveAsFlow()

    /** Builds the configuration from the current state, or null if it is not startable. */
    fun buildTrackingConfig(): TrackingConfig? {
        if (startProblem != null) return null
        val trackers = _trackerSlots.mapNotNull { slot ->
            slot.color?.let { ColorTrackerConfig(color = it, tolerance = slot.tolerance) }
        }
        return TrackingConfig(
            roi = roi,
            ruler = (rulerState as? RulerState.Valid)?.ruler,
            trackers = trackers,
            samplingRate = samplingRate ?: return null,
            zoomRatio = pipeline.zoomRatio.value
        )
    }

    fun startExperiment() {
        if (isStarting) return
        val config = buildTrackingConfig() ?: return
        val description = description.trim()
        isStarting = true
        startError = null
        viewModelScope.launch {
            try {
                val experimentId = db.write {
                    addExperiment(config, description, System.currentTimeMillis())
                }
                eventsChannel.send(UiEvent.ToLiveExperiment(experimentId))
            } catch (e: DatabaseException) {
                startError = e
                isStarting = false
            }
        }
    }

    init {
        viewModelScope.launch {
            snapshotFlow { roi }.collect { pipeline.roi = it }
        }
    }

    override fun onCleared() {
        super.onCleared()
        pipeline.close()
        tracker.close()
        eventsChannel.close()
    }

    companion object {
        /** Minimum width and height of a region of interest, as a fraction of the frame. */
        const val MIN_ROI_SIZE = 0.05f
    }
}
