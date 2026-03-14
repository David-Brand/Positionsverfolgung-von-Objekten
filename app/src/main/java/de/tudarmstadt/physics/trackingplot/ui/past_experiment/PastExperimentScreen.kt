package de.tudarmstadt.physics.trackingplot.ui.past_experiment

import android.content.res.Configuration
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.AxisBase
import com.github.mikephil.charting.formatter.ValueFormatter
import de.tudarmstadt.physics.trackingplot.tracker2.TrackingConfig
import kotlinx.coroutines.launch
import java.util.Locale
import kotlin.math.sqrt

@Composable
fun PastExperimentScreen(
    viewModel: PastExperimentViewModel
) {
    Column(
        modifier = Modifier
            .safeDrawingPadding()
            .fillMaxSize(),
//        contentAlignment = Alignment.Center
    ) {
        val context = LocalContext.current

        if (!viewModel.loaded) return

        val trackingConfig = viewModel.trackingConfig!!

        val chart = remember {
            LineChart(context).apply {
                axisRight.isEnabled = false
                description.isEnabled = false

                val timeFormatter = object : ValueFormatter() {
                    override fun getAxisLabel(value: Float, axis: AxisBase?): String? {
                        val millis = value.toLong()

                        val totalSeconds = millis / 1000
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
                xAxis.valueFormatter = timeFormatter

                trackingConfig.ruler?.let { ruler ->
                    val lengthFormatter = object : ValueFormatter() {
                        private val scale: Float
                        private val unitLabel = ruler.unit.symbol

                        init {
                            val dx = ruler.p2.x - ruler.p1.x
                            val dy = ruler.p2.y - ruler.p1.y

                            val normalizedDistance = sqrt(dx*dx + dy*dy)
                            scale = ruler.realDistance / normalizedDistance
                        }

                        override fun getAxisLabel(
                            value: Float,
                            axis: AxisBase?
                        ): String {
                            val realValue = value * scale
                            return String.format(Locale.getDefault(), "%.2f %s", realValue, unitLabel)
                        }
                    }

                    axisLeft.valueFormatter = lengthFormatter
                }

                data = viewModel.data
            }
        }

//        val loaded = viewModel.loaded
//        LaunchedEffect(loaded) {
//            if (loaded) {
//                chart.notifyDataSetChanged()
//                chart.invalidate()
//            }
//        }
        LaunchedEffect(Unit) {
            chart.notifyDataSetChanged()
            chart.invalidate()
        }

        val scope = rememberCoroutineScope()

        val createDocumentLauncher = rememberLauncherForActivityResult(
            ActivityResultContracts.CreateDocument("text/csv")
        ) { uri ->
            uri ?: return@rememberLauncherForActivityResult

            context.contentResolver.openOutputStream(uri)?.use { os ->
                os.write(viewModel.csvContent.toByteArray())
            }
        }

        val configuration = LocalConfiguration.current
        val isLandscape = configuration.orientation == Configuration.ORIENTATION_LANDSCAPE

        if (isLandscape) {
           Row(
               modifier = Modifier
                   .fillMaxSize()
                   .safeDrawingPadding()
           ) {
               Chart(
                   chart = chart,
                   trackingConfig = trackingConfig,
                   disabledLabels = viewModel.disabledLabels,
                   onToggleLabel = viewModel::toggleLabel,
                   modifier = Modifier
                       .fillMaxHeight()
                       .weight(1f)
               )
               Column(
                   modifier = Modifier.fillMaxHeight()
               ) {
                   Button(onClick = {
                       scope.launch {
                           viewModel.exportExperiment()
                           createDocumentLauncher.launch("experiment_${viewModel.experimentId}.csv")
                       }
                   }) {
                       Text("Export CSV")
                   }
                   Button(onClick = {
//                       scope.launch {
//                           viewModel.exportExperiment()
//                           createDocumentLauncher.launch("experiment_${viewModel.experimentId}.csv")
//                       }
                   }) {
                       Text("Delete")
                   }
               }
           }
        } else {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .safeDrawingPadding()
            ) {
                Chart(
                    chart = chart,
                    trackingConfig = viewModel.trackingConfig!!,
                    disabledLabels = viewModel.disabledLabels,
                    onToggleLabel = viewModel::toggleLabel,
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                )
                Row(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Button(onClick = {
                        scope.launch {
                            viewModel.exportExperiment()
                            createDocumentLauncher.launch("experiment_${viewModel.experimentId}.csv")
                        }
                    }) {
                        Text("Export CSV")
                    }
                    Button(onClick = {
//                       scope.launch {
//                           viewModel.exportExperiment()
//                           createDocumentLauncher.launch("experiment_${viewModel.experimentId}.csv")
//                       }
                    }) {
                        Text("Delete")
                    }
                }
            }
        }
    }
}

@Composable
private fun Chart(
    chart: LineChart,
    trackingConfig: TrackingConfig,
    disabledLabels: Set<String>,
    onToggleLabel: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
    ) {
        AndroidView(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            factory = { chart }
        )
        Row {
            trackingConfig.trackers.forEachIndexed { index, config ->
                val x = "${index}_X"
                val y = "${index}_Y"
                val xSelected = !disabledLabels.contains(x)
                val ySelected = !disabledLabels.contains(y)
                FilterChip(
                    selected = xSelected,
                    onClick = {
                        onToggleLabel(x)
                        chart.notifyDataSetChanged()
                        chart.invalidate()
                    },
                    label = { Text(text = x) },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = MaterialTheme.colorScheme.primaryContainer,
                        selectedLabelColor = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                )
                FilterChip(
                    selected = ySelected,
                    onClick = {
                        onToggleLabel(x)
                        chart.notifyDataSetChanged()
                        chart.invalidate()
                    },
                    label = { Text(text = y) },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = MaterialTheme.colorScheme.primaryContainer,
                        selectedLabelColor = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                )
            }
        }
    }
}