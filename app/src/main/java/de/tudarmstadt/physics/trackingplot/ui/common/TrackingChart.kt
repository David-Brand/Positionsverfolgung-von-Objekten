package de.tudarmstadt.physics.trackingplot.ui.common

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.horizontalScroll
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.AxisBase
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.formatter.ValueFormatter
import java.util.Locale

/**
 * Line chart of a [ChartModel] with a row of chips to show/hide individual series.
 *
 * @param axisWindow x-axis range (milliseconds) to display, or null to fit all data.
 */
@Composable
fun TrackingChart(
    model: ChartModel,
    modifier: Modifier = Modifier,
    axisWindow: ClosedFloatingPointRange<Float>? = null,
    trackerCount: Int = model.trackerIds.size
) {
    Column(modifier = modifier) {
        LineChartView(
            model = model,
            axisWindow = axisWindow,
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
        )
        SeriesChips(model = model, trackerCount = trackerCount)
    }
}

@Composable
private fun LineChartView(
    model: ChartModel,
    axisWindow: ClosedFloatingPointRange<Float>?,
    modifier: Modifier = Modifier
) {
    val colorScheme = MaterialTheme.colorScheme
    val textColor = colorScheme.onSurface.toArgb()
    val gridColor = colorScheme.outlineVariant.toArgb()
    val unitLabel = model.unitLabel

    AndroidView(
        modifier = modifier,
        factory = { context ->
            LineChart(context).apply {
                description.isEnabled = false
                axisRight.isEnabled = false
                setNoDataText("")
                setTouchEnabled(true)
                isDragEnabled = true
                setScaleEnabled(true)
                setPinchZoom(false)
                xAxis.position = XAxis.XAxisPosition.BOTTOM
                xAxis.valueFormatter = TimeAxisFormatter()
                axisLeft.valueFormatter = ValueAxisFormatter(unitLabel)
                legend.isWordWrapEnabled = true
            }
        },
        update = { chart ->
            // Reading the version here re-runs this block whenever the model commits.
            val version = model.version
            if (chart.data !== model.data) {
                chart.data = model.data
            }
            chart.xAxis.textColor = textColor
            chart.xAxis.gridColor = gridColor
            chart.xAxis.axisLineColor = gridColor
            chart.axisLeft.textColor = textColor
            chart.axisLeft.gridColor = gridColor
            chart.axisLeft.axisLineColor = gridColor
            chart.legend.textColor = textColor

            if (axisWindow != null) {
                chart.xAxis.axisMinimum = axisWindow.start
                chart.xAxis.axisMaximum = axisWindow.endInclusive
            } else {
                chart.xAxis.resetAxisMinimum()
                chart.xAxis.resetAxisMaximum()
            }
            chart.tag = version
            chart.notifyDataSetChanged()
            chart.invalidate()
        }
    )
}

@Composable
private fun SeriesChips(model: ChartModel, trackerCount: Int) {
    val ids = if (trackerCount > 0) (0 until trackerCount).toList() else model.trackerIds
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState())
            .padding(horizontal = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        for (trackerId in ids) {
            SeriesChip(model, ChartModel.xLabel(trackerId), trackerId)
            SeriesChip(model, ChartModel.yLabel(trackerId), trackerId)
        }
    }
}

@Composable
private fun SeriesChip(model: ChartModel, label: String, trackerId: Int) {
    val selected = model.isSeriesVisible(label)
    val trackerColor = TrackerColors.color(trackerId)
    FilterChip(
        selected = selected,
        onClick = { model.toggleSeries(label) },
        label = { Text(label) },
        colors = FilterChipDefaults.filterChipColors(
            selectedContainerColor = trackerColor.copy(alpha = 0.25f),
            selectedLabelColor = MaterialTheme.colorScheme.onSurface
        )
    )
}

/** Formats milliseconds as m:ss or h:mm:ss. */
private class TimeAxisFormatter : ValueFormatter() {
    override fun getAxisLabel(value: Float, axis: AxisBase?): String {
        val totalSeconds = (value / 1000f).toLong()
        val seconds = totalSeconds % 60
        val minutes = (totalSeconds / 60) % 60
        val hours = totalSeconds / 3600
        return if (hours > 0) {
            String.format(Locale.getDefault(), "%d:%02d:%02d", hours, minutes, seconds)
        } else {
            String.format(Locale.getDefault(), "%d:%02d", minutes, seconds)
        }
    }
}

private class ValueAxisFormatter(private val unitLabel: String?) : ValueFormatter() {
    override fun getAxisLabel(value: Float, axis: AxisBase?): String =
        if (unitLabel == null) {
            String.format(Locale.getDefault(), "%.3f", value)
        } else {
            String.format(Locale.getDefault(), "%.2f %s", value, unitLabel)
        }
}
