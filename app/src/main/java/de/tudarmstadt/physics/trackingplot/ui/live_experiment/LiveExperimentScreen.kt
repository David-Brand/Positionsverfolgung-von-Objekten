package de.tudarmstadt.physics.trackingplot.ui.live_experiment

import android.content.res.Configuration
import androidx.activity.compose.BackHandler
import androidx.camera.compose.CameraXViewfinder
import androidx.camera.core.CameraSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.lifecycle.awaitInstance
import androidx.camera.viewfinder.core.ImplementationMode
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.AxisBase
import com.github.mikephil.charting.formatter.ValueFormatter
import de.tudarmstadt.physics.trackingplot.R
import de.tudarmstadt.physics.trackingplot.tracker2.TrackingConfig
import kotlinx.coroutines.delay
import java.util.Locale
import kotlin.math.sqrt

enum class PanelMode {
    CAMERA,
    OPTIONS,
    PLOT_ONLY
}

@Composable
fun LiveExperimentScreen(
    onAbortExperiment: () -> Unit,
    viewModel: LiveExperimentViewModel
) {
    var showAbortDialog by remember { mutableStateOf(false) }

    BackHandler(enabled = true) {
        showAbortDialog = true
    }

    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val selector = CameraSelector.DEFAULT_BACK_CAMERA

    LaunchedEffect(Unit) {
        val provider = ProcessCameraProvider.awaitInstance(context)

        val useCaseGroup = viewModel.createUseCaseGroup()

        provider.unbindAll()
        provider.bindToLifecycle(lifecycleOwner, selector, useCaseGroup)
    }

    val surfaceRequest by viewModel.surfaceRequests.collectAsStateWithLifecycle()

    surfaceRequest?.let {
        val trackingConfig = viewModel.trackingConfig
            ?: return@let
        val disabledLines = viewModel.disabledLines

        val context = LocalContext.current
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
        LaunchedEffect(chart) {
            while (true) {
                delay(200)
                if (!chart.isEmpty) {
//                    chart.fitScreen()
                    chart.notifyDataSetChanged()
                    chart.invalidate()
                }
            }
        }

        val configuration = LocalConfiguration.current
        val isLandscape = configuration.orientation == Configuration.ORIENTATION_LANDSCAPE
        if (isLandscape) {
            var panelMode by rememberSaveable { mutableStateOf(PanelMode.CAMERA) }
            Row(
                modifier = Modifier
                    .fillMaxSize()
                    .safeDrawingPadding()
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxHeight()
                        .weight(1f)
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
                            val xSelected = !disabledLines.contains(x)
                            val ySelected = !disabledLines.contains(y)
                            FilterChip(
                                selected = xSelected,
                                onClick = {
                                    viewModel.toggleLine(x)
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
                                    viewModel.toggleLine(y)
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
                when (panelMode) {
                    PanelMode.CAMERA -> {
                        BoxWithConstraints(
                            modifier = Modifier
                                .fillMaxHeight()
                                .weight(1f),
//                                .background(Color.Green),
                            contentAlignment = Alignment.TopCenter
                        ) {
                            val aspectRatio = 3 to 4
                            val maxAspectRatio: Float = maxWidth / maxHeight
                            val aspectRatioFloat: Float =
                                aspectRatio.first.toFloat() / aspectRatio.second
                            val shouldUseMaxWidth = maxAspectRatio <= aspectRatioFloat
                            var width =
                                if (shouldUseMaxWidth) maxWidth else maxHeight * aspectRatioFloat
                            var height =
                                if (!shouldUseMaxWidth) maxHeight else maxWidth / aspectRatioFloat

                            Box(
                                modifier = Modifier
                                    .width(width)
                                    .height(height)
                            ) {
                                val implementationMode = ImplementationMode.EXTERNAL
                                CameraXViewfinder(
                                    modifier = Modifier
                                        .fillMaxSize(),
                                    surfaceRequest = it,
                                    implementationMode = implementationMode,
                                    contentScale = ContentScale.Fit
                                )

                                val boxes by viewModel.boxes.collectAsStateWithLifecycle()
                                val boundingBoxPoints = viewModel.boundingBoxPoints
                                Canvas(
                                    modifier = Modifier.matchParentSize()
                                ) {
                                    val canvasWidth = size.width
                                    val canvasHeight = size.height

                                    boxes.forEach { box ->

                                        drawRect(
                                            color = Color.Red,
                                            topLeft = Offset(
                                                box.left * canvasWidth,
                                                box.top * canvasHeight
                                            ),
                                            size = Size(
                                                (box.right - box.left) * canvasWidth,
                                                (box.bottom - box.top) * canvasHeight
                                            ),
                                            style = Stroke(width = 4f)
                                        )
                                    }


                                    boundingBoxPoints?.let { (first, second) ->
                                        val topLeft = Offset(
                                            minOf(first.x, second.x) * canvasWidth,
                                            minOf(first.y, second.y) * canvasHeight
                                        )
                                        val bottomRight = Offset(
                                            maxOf(first.x, second.x) * canvasWidth,
                                            maxOf(first.y, second.y) * canvasHeight
                                        )

                                        drawRect(
                                            color = Color.Green.copy(alpha = 0.5f),
                                            topLeft = topLeft,
                                            size = Size(
                                                width = bottomRight.x - topLeft.x,
                                                height = bottomRight.y - topLeft.y
                                            ),
                                            style = Stroke(
                                                width = 2.dp.toPx(),
                                                cap = StrokeCap.Round,
                                                join = StrokeJoin.Round
                                            )
                                        )
                                    }
                                }
                            }
                        }
                    }
                    PanelMode.OPTIONS -> {
                        val timeWindowMs = viewModel.timeWindowMs
                        val offsetMs = viewModel.offsetMs
                        Column(
                            modifier = Modifier
                                .fillMaxHeight()
                                .weight(1f),
                        ) {
                            Text("Time window: ${timeWindowMs / 1000}s")
                            Slider(
                                value = viewModel.timeWindowSlider,
                                onValueChange = viewModel::onTimeWindowSliderValueChange
                            )

                            Spacer(Modifier.height(8.dp))

                            Text(
                                if (offsetMs == 0L) "Live"
                                else "Offset: ${offsetMs / 1000}s"
                            )
                            Slider(
                                value = viewModel.offsetSlider,
                                onValueChange = viewModel::onOffsetSliderValueChange
                            )
                        }
                    }
                    PanelMode.PLOT_ONLY -> {}
                }
                Column(
                    modifier = Modifier.fillMaxHeight(),
                    verticalArrangement = Arrangement.Center
                ) {
                    IconButton(onClick = {
                        panelMode = PanelMode.PLOT_ONLY
                    }) {
                        Icon(
                            painter = painterResource(id = R.drawable.baseline_show_chart_24),
                            contentDescription = null
                        )
                    }
                    IconButton(onClick = {
                        panelMode = PanelMode.CAMERA
                    }) {
                        Icon(
                            painter = painterResource(id = R.drawable.baseline_photo_camera_24),
                            contentDescription = null
                        )
                    }
                    IconButton(onClick = {
                        panelMode = PanelMode.OPTIONS
                    }) {
                        Icon(
                            painter = painterResource(id = R.drawable.baseline_control_camera_24),
                            contentDescription = null
                        )
                    }

                    Spacer(modifier = Modifier.weight(1f))

                    if (viewModel.isRecording) {
                        Button(
//                            onClick = { showAbortDialog = true },
                            onClick = { viewModel.toggleRecording(context) },
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color(0xFFD32F2F)
                            )
                        ) {
                            Text("Finish")
                        }
                    } else {
                        Button(
                            onClick = { viewModel.toggleRecording(context) },
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color(0xFF689F38)
                            )
                        ) {
                            Text("Start")
                        }
                    }
                }
            }
        } else {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .safeDrawingPadding()
            ) {
                BoxWithConstraints(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    contentAlignment = Alignment.TopCenter
                ) {
                    val aspectRatio = 3 to 4
                    val maxAspectRatio: Float = maxWidth / maxHeight
                    val aspectRatioFloat: Float = aspectRatio.first.toFloat() / aspectRatio.second
                    val shouldUseMaxWidth = maxAspectRatio <= aspectRatioFloat
                    var width = if (shouldUseMaxWidth) maxWidth else maxHeight * aspectRatioFloat
                    var height = if (!shouldUseMaxWidth) maxHeight else maxWidth / aspectRatioFloat

                    //todo temp
                    width /= 2
                    height /= 2

                    Box(
                        modifier = Modifier
                            .width(width)
                            .height(height)
                    ) {
                        val implementationMode = ImplementationMode.EXTERNAL
                        CameraXViewfinder(
                            modifier = Modifier
                                .fillMaxSize(),
                            surfaceRequest = it,
                            implementationMode = implementationMode,
                            contentScale = ContentScale.Fit
                        )

                        val boxes by viewModel.boxes.collectAsStateWithLifecycle()
                        val boundingBoxPoints = viewModel.boundingBoxPoints
                        Canvas(
                            modifier = Modifier.matchParentSize()
                        ) {
                            val canvasWidth = size.width
                            val canvasHeight = size.height

                            boxes.forEach { box ->

                                drawRect(
                                    color = Color.Red,
                                    topLeft = Offset(
                                        box.left * canvasWidth,
                                        box.top * canvasHeight
                                    ),
                                    size = Size(
                                        (box.right - box.left) * canvasWidth,
                                        (box.bottom - box.top) * canvasHeight
                                    ),
                                    style = Stroke(width = 4f)
                                )
                            }


                            boundingBoxPoints?.let { (first, second) ->
                                val topLeft = Offset(
                                    minOf(first.x, second.x) * canvasWidth,
                                    minOf(first.y, second.y) * canvasHeight
                                )
                                val bottomRight = Offset(
                                    maxOf(first.x, second.x) * canvasWidth,
                                    maxOf(first.y, second.y) * canvasHeight
                                )

                                drawRect(
                                    color = Color.Green.copy(alpha = 0.5f),
                                    topLeft = topLeft,
                                    size = Size(
                                        width = bottomRight.x - topLeft.x,
                                        height = bottomRight.y - topLeft.y
                                    ),
                                    style = Stroke(
                                        width = 2.dp.toPx(),
                                        cap = StrokeCap.Round,
                                        join = StrokeJoin.Round
                                    )
                                )
                            }
                        }
                    }

                    Column(
                        modifier = Modifier
                            .align(Alignment.BottomCenter)
                            .fillMaxWidth()
                            .height(maxHeight - height)
                    ) {
                        AndroidView(
                            modifier = Modifier
                                .fillMaxWidth()
                                .weight(1f),
                            factory = { chart }
                        )
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            trackingConfig.trackers.forEachIndexed { index, config ->
                                val x = "${index}_X"
                                val y = "${index}_Y"
                                val xSelected = !disabledLines.contains(x)
                                val ySelected = !disabledLines.contains(y)
                                FilterChip(
                                    selected = xSelected,
                                    onClick = {
                                        viewModel.toggleLine(x)
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
                                        viewModel.toggleLine(y)
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

                if (viewModel.isRecording) {
                    Button(
                        onClick = { showAbortDialog = true },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFFD32F2F)
                        )
                    ) {
                        Text("Finish")
                    }
                } else {
                    Button(
                        onClick = { viewModel.toggleRecording(context) },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF689F38)
                        )
                    ) {
                        Text("Start")
                    }
                }
            }
        }
    }

    if (showAbortDialog) {
        AlertDialog(
            onDismissRequest = { showAbortDialog = false },
            title = { Text(stringResource(R.string.abort_experiment)) },
            text = { Text("Gathered data will be saved, but no further data is collected") },
            confirmButton = {
                TextButton(onClick = {
                    showAbortDialog = false
                    onAbortExperiment()
                }) {
                    Text("Abort", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { showAbortDialog = false }) {
                    Text("Continue")
                }
            }
        )
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