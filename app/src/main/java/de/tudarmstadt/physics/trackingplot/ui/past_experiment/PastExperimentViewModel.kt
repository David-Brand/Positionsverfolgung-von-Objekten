package de.tudarmstadt.physics.trackingplot.ui.past_experiment

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.mutableStateSetOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import de.tudarmstadt.physics.trackingplot.db.ExperimentDatabase
import de.tudarmstadt.physics.trackingplot.tracker2.ColorTrackerConfig
import de.tudarmstadt.physics.trackingplot.tracker2.TrackingConfig
import kotlinx.coroutines.launch

class PastExperimentViewModel(
    val experimentId: Long,
    private val db: ExperimentDatabase
): ViewModel() {

    var csvContent = ""

    suspend fun deleteExperiment() {
        db.withTransaction(readOnly = false) {
            deleteExperiment(experimentId)
        }
    }

    suspend fun exportExperiment() {
        val header = "trackerIndex,timestamp,centroidX,centroidY"
        val measurements = db.withTransaction(readOnly = true) {
            getExperimentMeasurements(experimentId)
        }
        val rows = measurements.joinToString("\n") { it }
        csvContent = "$header\n$rows"
    }

    val data = LineData()

    private val _trackingConfig = mutableStateOf<TrackingConfig?>(null)
    val trackingConfig by _trackingConfig
    private val _disabledLabels = mutableStateSetOf<String>()
    val disabledLabels = _disabledLabels as Set<String>

    private val _loaded = mutableStateOf(false)
    val loaded by _loaded

    init {
        viewModelScope.launch {
            val pastData = db.withTransaction(readOnly = true) {
                getPastData(experimentId)
            }
            pastData.forEach { (trackerIndex, measurements) ->
                val color = when (trackerIndex) {
                    0 -> Color.Red
                    1 -> Color.Green
                    2 -> Color.Blue
                    else -> Color.Gray
                }.toArgb()

                val labelX = "${trackerIndex}_X"
                val labelY = "${trackerIndex}_Y"
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

                for ((timestamp, centroid) in measurements) {
                    val x = centroid.centroidX.toFloat()
                    val y = centroid.centroidY.toFloat()
                    dataSetX.addEntry(Entry(timestamp.toFloat(), x))
                    dataSetY.addEntry(Entry(timestamp.toFloat(), y))
                }

                dataSetX.notifyDataSetChanged()
                dataSetY.notifyDataSetChanged()
            }

            val trackingConfig = db.withTransaction(readOnly = true) {
                getTrackingConfig(experimentId)
            }

            _trackingConfig.value = trackingConfig

            data.notifyDataChanged()

            _loaded.value = true
        }
    }

    fun toggleLabel(label: String) {
        val isCurrentlyVisible = !_disabledLabels.contains(label)

        val dataSet = data.getDataSetByLabel(label, true) as? LineDataSet
            ?: return

        if (isCurrentlyVisible) {
            dataSet.isVisible = false
            _disabledLabels.add(label)
        } else {
            dataSet.isVisible = true
            _disabledLabels.remove(label)
        }

        dataSet.notifyDataSetChanged()

        data.notifyDataChanged()
    }
}