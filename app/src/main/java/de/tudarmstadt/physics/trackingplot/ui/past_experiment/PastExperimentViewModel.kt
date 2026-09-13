package de.tudarmstadt.physics.trackingplot.ui.past_experiment

import android.content.ContentResolver
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import de.tudarmstadt.physics.trackingplot.db.DatabaseException
import de.tudarmstadt.physics.trackingplot.db.ExperimentDatabase
import de.tudarmstadt.physics.trackingplot.geometry.RulerScale
import de.tudarmstadt.physics.trackingplot.model.ExperimentSummary
import de.tudarmstadt.physics.trackingplot.model.MeasurementSample
import de.tudarmstadt.physics.trackingplot.model.TrackingConfig
import de.tudarmstadt.physics.trackingplot.ui.common.ChartModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class PastExperimentViewModel(
    val experimentId: Long,
    private val db: ExperimentDatabase
) : ViewModel() {

    sealed interface State {
        data object Loading : State
        data class Failed(val error: Throwable?) : State
        data class Ready(
            val summary: ExperimentSummary,
            val config: TrackingConfig,
            val chart: ChartModel
        ) : State
    }

    sealed interface ExportState {
        data object Idle : ExportState
        data object Exporting : ExportState
        data object Done : ExportState
        data class Failed(val error: Throwable) : ExportState
    }

    sealed interface DeleteState {
        data object Idle : DeleteState
        data object Deleting : DeleteState
        data object Done : DeleteState
        data class Failed(val error: Throwable) : DeleteState
    }

    private val _state = MutableStateFlow<State>(State.Loading)
    val state: StateFlow<State> = _state

    private val _exportState = MutableStateFlow<ExportState>(ExportState.Idle)
    val exportState: StateFlow<ExportState> = _exportState

    private val _deleteState = MutableStateFlow<DeleteState>(DeleteState.Idle)
    val deleteState: StateFlow<DeleteState> = _deleteState

    /** Suggested file name for the CSV export. */
    val csvFileName: String get() = "experiment_$experimentId.csv"

    private var measurements: Map<Int, List<MeasurementSample>> = emptyMap()

    init {
        load()
    }

    fun reload() = load()

    private fun load() {
        _state.value = State.Loading
        viewModelScope.launch {
            try {
                val ready = db.read {
                    val summary = getExperiment(experimentId) ?: return@read null
                    val config = getTrackingConfig(experimentId) ?: return@read null
                    val samples = getMeasurements(experimentId)
                    val chart = ChartModel(RulerScale.from(config.ruler))
                    samples.forEach { (trackerId, list) ->
                        list.forEach { chart.addSample(trackerId, it.timestampMs, it.centroidX, it.centroidY) }
                    }
                    chart.commit()
                    measurements = samples
                    State.Ready(summary, config, chart)
                }
                _state.value = ready ?: State.Failed(null)
            } catch (e: DatabaseException) {
                _state.value = State.Failed(e)
            }
        }
    }

    /** Writes the measurements as CSV to [uri] (obtained from a CreateDocument request). */
    fun exportTo(uri: Uri, contentResolver: ContentResolver) {
        val ready = _state.value as? State.Ready ?: return
        if (_exportState.value == ExportState.Exporting) return
        _exportState.value = ExportState.Exporting
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val csv = buildCsv(ready.config, measurements)
                    val stream = contentResolver.openOutputStream(uri, "wt")
                        ?: throw IllegalStateException("Could not open $uri for writing")
                    stream.use { it.write(csv.toByteArray(Charsets.UTF_8)) }
                }
                _exportState.value = ExportState.Done
            } catch (e: Exception) {
                _exportState.value = ExportState.Failed(e)
            }
        }
    }

    fun dismissExportState() {
        _exportState.value = ExportState.Idle
    }

    fun delete() {
        if (_deleteState.value == DeleteState.Deleting) return
        _deleteState.value = DeleteState.Deleting
        viewModelScope.launch {
            try {
                db.write { deleteExperiment(experimentId) }
                _deleteState.value = DeleteState.Done
            } catch (e: DatabaseException) {
                _deleteState.value = DeleteState.Failed(e)
            }
        }
    }

    fun dismissDeleteState() {
        if (_deleteState.value is DeleteState.Failed) _deleteState.value = DeleteState.Idle
    }

    companion object {
        /**
         * CSV layout: the first four columns are unchanged from earlier versions
         * (trackerIndex, timestamp in ms, normalised centroid x/y); when the experiment has a ruler,
         * two columns with the position in real-world units follow.
         */
        fun buildCsv(config: TrackingConfig, measurements: Map<Int, List<MeasurementSample>>): String {
            val scale = RulerScale.from(config.ruler)
            val builder = StringBuilder()
            builder.append("trackerIndex,timestamp,centroidX,centroidY")
            if (scale != null) {
                builder.append(",x_").append(scale.unit.symbol).append(",y_").append(scale.unit.symbol)
            }
            builder.append('\n')
            for (trackerIndex in measurements.keys.sorted()) {
                for (sample in measurements.getValue(trackerIndex)) {
                    builder.append(sample.trackerIndex).append(',')
                        .append(sample.timestampMs).append(',')
                        .append(sample.centroidX).append(',')
                        .append(sample.centroidY)
                    if (scale != null) {
                        builder.append(',').append(scale.x(sample.centroidX))
                            .append(',').append(scale.y(sample.centroidY))
                    }
                    builder.append('\n')
                }
            }
            return builder.toString()
        }
    }
}
