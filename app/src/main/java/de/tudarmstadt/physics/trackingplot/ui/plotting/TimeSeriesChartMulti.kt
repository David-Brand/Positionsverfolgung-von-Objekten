package de.tudarmstadt.physics.trackingplot.ui.plotting

import android.graphics.Color
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet

@Composable
fun TimeSeriesChartMulti(
    xPoints: List<TimePoint>,
    yPoints: List<TimePoint>,
    visibleWindowMs: Long,
    windowOffsetMs: Long,
    modifier: Modifier = Modifier
) {
    AndroidView(
        modifier = modifier,
        factory = { context ->
            LineChart(context).apply {
                axisRight.isEnabled = false
                description.isEnabled = false
                data = LineData()
            }
        },
        update = { chart ->
            updateChart(
                chart = chart,
                xPoints = xPoints,
                yPoints = yPoints,
                visibleWindowMs = visibleWindowMs,
                windowOffsetMs = windowOffsetMs
            )
//            val data = chart.data
//            data.clearValues()
//
//            fun normalizeY(y: Float): Float {
//                return 1f - y  // invert so top=1 -> top on plot
//            }
//
//            fun addSet(points: List<TimePoint>, label: String, color: Int) {
//                if (points.isEmpty()) return
//                val base = points.first().timeMs
//                val entries = points.map {
//                    Entry(
//                        (it.timeMs - base).toFloat(),
//                        normalizeY(it.value)
//                    )
//                }
//                data.addDataSet(
//                    LineDataSet(entries, label).apply {
//                        setDrawCircles(false)
//                        this.color = color
//                    }
//                )
//            }
//
//            addSet(xPoints, "X", Color.RED)
//            addSet(yPoints, "Y", Color.BLUE)
//
//            chart.notifyDataSetChanged()
//            chart.invalidate()
        }
    )
}

fun updateChart(
    chart: LineChart,
    xPoints: List<TimePoint>,
    yPoints: List<TimePoint>,
    visibleWindowMs: Long,
    windowOffsetMs: Long
) {
    chart.data.clearValues()

    if (xPoints.isEmpty() && yPoints.isEmpty()) return

    val data = chart.data ?: LineData().also { chart.data = it }

    fun normalizeY(y: Float): Float = 1f - y  // invert Y

    fun updateDataSet(points: List<TimePoint>, label: String, color: Int) {
        if (points.isEmpty()) return

        val dataSet = data.getDataSetByLabel(label, true) as? LineDataSet
            ?: LineDataSet(mutableListOf(), label).apply {
                setDrawCircles(false)
                this.color = color
                data.addDataSet(this)
            }

        val baseTime = points.first().timeMs

        // Append only new points
        val existingCount = dataSet.entryCount
        for (i in existingCount until points.size) {
            val p = points[i]
            val x = (p.timeMs - baseTime).toFloat()
            val y = normalizeY(p.value)
            dataSet.addEntry(Entry(x, y))
        }

        dataSet.notifyDataSetChanged()
    }

    updateDataSet(xPoints, "X", Color.RED)
    updateDataSet(yPoints, "Y", Color.BLUE)

    data.notifyDataChanged()
    chart.notifyDataSetChanged()

    // Compute sliding window
    val allPoints = (xPoints + yPoints).sortedBy { it.timeMs }
    if (allPoints.isNotEmpty()) {
        val baseTime = allPoints.first().timeMs
        val totalDuration = allPoints.last().timeMs - baseTime

        val windowEnd = (totalDuration - windowOffsetMs).coerceAtLeast(0)
        val windowStart = (windowEnd - visibleWindowMs).coerceAtLeast(0)

        chart.xAxis.axisMinimum = windowStart.toFloat()
        chart.xAxis.axisMaximum = windowEnd.toFloat()
    }

    chart.invalidate()
}

