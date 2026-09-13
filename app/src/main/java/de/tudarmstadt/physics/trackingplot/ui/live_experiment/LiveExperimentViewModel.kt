package de.tudarmstadt.physics.trackingplot.ui.live_experiment

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import de.tudarmstadt.physics.trackingplot.camera.CameraPipeline
import de.tudarmstadt.physics.trackingplot.db.DatabaseException
import de.tudarmstadt.physics.trackingplot.db.ExperimentDatabase
import de.tudarmstadt.physics.trackingplot.db.MeasurementWriter
import de.tudarmstadt.physics.trackingplot.geometry.RulerScale
import de.tudarmstadt.physics.trackingplot.model.ExperimentSummary
import de.tudarmstadt.physics.trackingplot.model.FrameResult
import de.tudarmstadt.physics.trackingplot.model.HsvColor
import de.tudarmstadt.physics.trackingplot.model.HsvTolerance
import de.tudarmstadt.physics.trackingplot.model.MeasurementSample
import de.tudarmstadt.physics.trackingplot.model.TrackedObject
import de.tudarmstadt.physics.trackingplot.model.TrackingConfig
import de.tudarmstadt.physics.trackingplot.tracker.NativeTracker
import de.tudarmstadt.physics.trackingplot.ui.common.ChartModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * Runs a stored experiment: configures the tracker from the persisted configuration, records
 * samples while [isRecording], plots them and persists them in the background.
 *
 * The time axis starts at the first recorded frame and keeps running across stop/start, so
 * samples never overlap. If the experiment already has samples (the process was restarted while
 * the screen was open), new samples continue after the last stored timestamp.
 */
class LiveExperimentViewModel(
    val experimentId: Long,
    private val db: ExperimentDatabase,
    appScope: CoroutineScope
) : ViewModel() {

    sealed interface LoadState {
        data object Loading : LoadState
        data class Failed(val error: Throwable?) : LoadState
        data class Ready(
            val config: TrackingConfig,
            val summary: ExperimentSummary,
            val chart: ChartModel
        ) : LoadState
    }

    private val _loadState = MutableStateFlow<LoadState>(LoadState.Loading)
    val loadState: StateFlow<LoadState> = _loadState

    private val tracker = NativeTracker()

    @Volatile
    private var analysisEnabled = false

    val pipeline = CameraPipeline(tracker) { analysisEnabled }

    private val writer = MeasurementWriter(db, experimentId, appScope)

    /** Set when samples could not be written to the database. */
    val writerError: StateFlow<Throwable?> = writer.error

    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording

    private val _trackedObjects = MutableStateFlow<List<TrackedObject>>(emptyList())

    /** Objects found in the most recent frame, whether recording or not. */
    val trackedObjects: StateFlow<List<TrackedObject>> = _trackedObjects

    private val _elapsedMs = MutableStateFlow(0L)

    /** Time of the last recorded sample, in milliseconds since the experiment start. */
    val elapsedMs: StateFlow<Long> = _elapsedMs

    private val _sampleCount = MutableStateFlow(0L)
    val sampleCount: StateFlow<Long> = _sampleCount

    // ---------------------------------------------------------------- chart viewport

    /** 0..1, mapped logarithmically to 1 s .. 6 h; 1 shows the whole experiment. */
    var timeWindowSlider by mutableFloatStateOf(1f)
        private set

    /** 0..1; 1 follows the newest data ("live"), smaller values look further into the past. */
    var offsetSlider by mutableFloatStateOf(1f)
        private set

    fun onTimeWindowSliderChange(value: Float) {
        timeWindowSlider = value.coerceIn(0f, 1f)
    }

    fun onOffsetSliderChange(value: Float) {
        offsetSlider = value.coerceIn(0f, 1f)
    }

    val timeWindowMs: Long get() = logarithmicTimeWindowMs(timeWindowSlider)

    fun offsetMs(latestTimeMs: Float): Long = ((1f - offsetSlider) * max(latestTimeMs, 0f)).toLong()

    /**
     * X-axis range to display for the current slider positions, or null to fit everything.
     */
    fun axisWindow(latestTimeMs: Float): ClosedFloatingPointRange<Float>? {
        if (latestTimeMs <= 0f) return null
        val window = timeWindowMs.toFloat()
        val end = latestTimeMs - offsetMs(latestTimeMs)
        if (window >= latestTimeMs && offsetSlider >= 1f) return null
        val start = max(0f, end - window)
        return start..max(end, start + 1f)
    }

    // ---------------------------------------------------------------- recording

    private var chart: ChartModel? = null
    private var baseTimestampNanos: Long? = null
    private var timeOffsetMs = 0L
    private var lastAcceptedNanos = Long.MIN_VALUE
    private var minIntervalNanos = 0L

    fun startRecording() {
        if (_loadState.value !is LoadState.Ready) return
        _isRecording.value = true
    }

    fun stopRecording() {
        _isRecording.value = false
    }

    fun toggleRecording() {
        if (_isRecording.value) stopRecording() else startRecording()
    }

    init {
        load()
        viewModelScope.launch(Dispatchers.Main.immediate) {
            pipeline.frames.collect { onFrame(it) }
        }
        viewModelScope.launch {
            while (isActive) {
                delay(CHART_REFRESH_MS)
                chart?.commitIfDirty()
            }
        }
    }

    private fun load() {
        viewModelScope.launch {
            try {
                val loaded = db.read {
                    val summary = getExperiment(experimentId) ?: return@read null
                    val config = getTrackingConfig(experimentId) ?: return@read null
                    Loaded(summary, config, getMeasurements(experimentId), getLastTimestamp(experimentId))
                }
                if (loaded == null) {
                    _loadState.value = LoadState.Failed(null)
                    return@launch
                }
                val chart = ChartModel(RulerScale.from(loaded.config.ruler))
                loaded.measurements.forEach { (trackerId, samples) ->
                    samples.forEach { chart.addSample(trackerId, it.timestampMs, it.centroidX, it.centroidY) }
                }
                chart.commit()

                timeOffsetMs = loaded.lastTimestampMs?.let { it + 1 } ?: 0L
                minIntervalNanos = if (loaded.config.samplingRate > 0) {
                    1_000_000_000L / loaded.config.samplingRate
                } else 0L
                _elapsedMs.value = loaded.lastTimestampMs ?: 0L
                _sampleCount.value = loaded.summary.sampleCount

                configureTracker(loaded.config)
                this@LiveExperimentViewModel.chart = chart
                _loadState.value = LoadState.Ready(loaded.config, loaded.summary, chart)
                analysisEnabled = true
            } catch (e: DatabaseException) {
                _loadState.value = LoadState.Failed(e)
            }
        }
    }

    private class Loaded(
        val summary: ExperimentSummary,
        val config: TrackingConfig,
        val measurements: Map<Int, List<MeasurementSample>>,
        val lastTimestampMs: Long?
    )

    private fun configureTracker(config: TrackingConfig) {
        tracker.clear()
        config.trackers.forEachIndexed { index, trackerConfig ->
            tracker.setTracker(
                index,
                HsvColor.fromArgb(trackerConfig.color),
                HsvTolerance.fromUiTolerance(trackerConfig.tolerance)
            )
        }
        pipeline.roi = config.roi
        pipeline.setZoomRatio(config.zoomRatio)
    }

    /** Called on the main thread for every analysed frame. */
    private fun onFrame(result: FrameResult) {
        _trackedObjects.value = result.objects
        if (!_isRecording.value) return
        val chart = chart ?: return

        val base = baseTimestampNanos ?: result.timestampNanos.also { baseTimestampNanos = it }
        val timestamp = result.timestampNanos
        if (minIntervalNanos > 0 && lastAcceptedNanos != Long.MIN_VALUE &&
            timestamp - lastAcceptedNanos < minIntervalNanos * 9 / 10
        ) {
            return
        }
        lastAcceptedNanos = timestamp

        val timeMs = timeOffsetMs + (timestamp - base) / 1_000_000L
        if (result.objects.isEmpty()) return

        val samples = result.objects.map { obj ->
            chart.addSample(obj.trackerId, timeMs, obj.centroidX, obj.centroidY)
            MeasurementSample(obj.trackerId, timeMs, obj.centroidX, obj.centroidY)
        }
        writer.submit(samples)
        _elapsedMs.value = timeMs
        _sampleCount.value += samples.size
    }

    override fun onCleared() {
        super.onCleared()
        _isRecording.value = false
        analysisEnabled = false
        pipeline.close()
        tracker.close()
        writer.close()
    }

    companion object {
        private const val CHART_REFRESH_MS = 100L
        const val MIN_TIME_WINDOW_MS = 1_000L
        const val MAX_TIME_WINDOW_MS = 6L * 60 * 60 * 1_000

        fun logarithmicTimeWindowMs(fraction: Float): Long {
            val f = fraction.coerceIn(0f, 1f).toDouble()
            val logMin = ln(MIN_TIME_WINDOW_MS.toDouble())
            val logMax = ln(MAX_TIME_WINDOW_MS.toDouble())
            return exp(logMin + f * (logMax - logMin)).toLong()
        }
    }
}
