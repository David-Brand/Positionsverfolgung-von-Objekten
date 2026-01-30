package de.tudarmstadt.physics.trackingplot.ui.plotting

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.formatter.ValueFormatter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

//@Composable
//fun TimeSeriesChart(
//    points: List<TimePoint>,
//    visibleWindowMs: Long,
//    windowOffsetMs: Long, // 0 = live, positive = move into past
//    modifier: Modifier = Modifier
//) {
//    AndroidView(
//        modifier = modifier,
//        factory = { context ->
//            LineChart(context).apply {
//                description.isEnabled = false
//                setTouchEnabled(true)
//                isDragEnabled = true
//                setScaleEnabled(false)
//                setPinchZoom(false)
//
//                axisRight.isEnabled = false
//
//                xAxis.apply {
//                    position = XAxis.XAxisPosition.BOTTOM
//                    setDrawGridLines(false)
//                    valueFormatter = object : ValueFormatter() {
//                        override fun getFormattedValue(value: Float): String {
//                            val time = value.toLong()
//                            return SimpleDateFormat("HH:mm:ss", Locale.getDefault())
//                                .format(Date(time))
//                        }
//                    }
//                }
//
//                axisLeft.setDrawGridLines(true)
//
//                data = LineData().apply {
//                    setDrawValues(false)
//                }
//            }
//        },
//        update = { chart ->
//            if (points.isEmpty()) return@AndroidView
//
//            val entries = points.map {
//                Entry(it.timeMs.toFloat(), it.value)
//            }
//
//            val dataSet = LineDataSet(entries, "Signal").apply {
//                setDrawCircles(false)
//                lineWidth = 1.5f
//                mode = LineDataSet.Mode.LINEAR
//            }
//
//            chart.data = LineData(dataSet)
//
//            val now = System.currentTimeMillis()
//            val windowEnd = now - windowOffsetMs
//            val windowStart = windowEnd - visibleWindowMs
//
//            chart.xAxis.axisMinimum = windowStart.toFloat()
//            chart.xAxis.axisMaximum = windowEnd.toFloat()
//
//            chart.invalidate()
//        }
//    )
//}

@Composable
fun TimeSeriesChart(
    points: List<TimePoint>,
    visibleWindowMs: Long,
    windowOffsetMs: Long,
    modifier: Modifier = Modifier
) {
    AndroidView(
        modifier = modifier,
        factory = { context ->
            val chart = LineChart(context)

            val dataSet = LineDataSet(mutableListOf(), "Signal").apply {
                setDrawCircles(false)
                lineWidth = 1.5f
                mode = LineDataSet.Mode.LINEAR
            }

            chart.apply {
                description.isEnabled = false
                axisRight.isEnabled = false
                setTouchEnabled(true)

                xAxis.apply {
                    position = XAxis.XAxisPosition.BOTTOM
                    setDrawGridLines(false)
                }

                data = LineData(dataSet).apply {
                    setDrawValues(false)
                }
            }

            chart
        },
        update = { chart ->
            if (points.isEmpty()) return@AndroidView

            val dataSet = chart.data.getDataSetByIndex(0) as LineDataSet
            val baseTime = points.first().timeMs

            // Append only new points
            val existingCount = dataSet.entryCount
            for (i in existingCount until points.size) {
                val p = points[i]
                val x = (p.timeMs - baseTime).toFloat()
                dataSet.addEntry(Entry(x, p.value))
            }

            chart.data.notifyDataChanged()
            chart.notifyDataSetChanged()

            val totalDuration = points.last().timeMs - baseTime
            val windowEnd = totalDuration - windowOffsetMs
            val windowStart = (windowEnd - visibleWindowMs).coerceAtLeast(0)

            chart.xAxis.axisMinimum = windowStart.toFloat()
            chart.xAxis.axisMaximum = windowEnd.toFloat()

            chart.invalidate()
        }
    )
}
