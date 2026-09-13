package de.tudarmstadt.physics.trackingplot.ui.live_experiment

import android.content.res.Configuration
import androidx.activity.compose.BackHandler
import androidx.annotation.DrawableRes
import androidx.annotation.StringRes
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import de.tudarmstadt.physics.trackingplot.R
import de.tudarmstadt.physics.trackingplot.camera.CameraPipeline
import de.tudarmstadt.physics.trackingplot.model.Point2D
import de.tudarmstadt.physics.trackingplot.model.Roi
import de.tudarmstadt.physics.trackingplot.model.Ruler
import de.tudarmstadt.physics.trackingplot.model.TrackedObject
import de.tudarmstadt.physics.trackingplot.ui.common.CameraFrameBox
import de.tudarmstadt.physics.trackingplot.ui.common.FrameMapper
import de.tudarmstadt.physics.trackingplot.ui.common.TrackerColors
import de.tudarmstadt.physics.trackingplot.ui.common.TrackingChart
import kotlin.math.roundToInt

/** Which panel the landscape layout shows next to the chart. */
private enum class PanelMode { PLOT_ONLY, CAMERA, OPTIONS }

private val RecordingRed = Color(0xFFD32F2F)
private val StartGreen = Color(0xFF2E7D32)
private val RoiGreen = Color(0xFF4CAF50)
private val RulerCyan = Color.Cyan

/** Largest share of the portrait height the camera preview may take. */
private const val PORTRAIT_CAMERA_MAX_HEIGHT_FRACTION = 0.45f
private val PortraitCameraMaxWidth = 360.dp

@Composable
fun LiveExperimentScreen(onLeave: () -> Unit, onShowResults: () -> Unit, viewModel: LiveExperimentViewModel) {
    val loadState by viewModel.loadState.collectAsStateWithLifecycle()
    val isRecording by viewModel.isRecording.collectAsStateWithLifecycle()
    var showLeaveDialog by rememberSaveable { mutableStateOf(false) }

    BackHandler {
        if (isRecording) showLeaveDialog = true else onLeave()
    }

    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var bindAttempt by remember { mutableIntStateOf(0) }
    LaunchedEffect(lifecycleOwner, bindAttempt) {
        viewModel.pipeline.bind(context, lifecycleOwner)
    }

    when (val state = loadState) {
        LiveExperimentViewModel.LoadState.Loading -> LoadingContent()
        is LiveExperimentViewModel.LoadState.Failed -> LoadFailedContent(error = state.error, onBack = onLeave)
        is LiveExperimentViewModel.LoadState.Ready -> ReadyContent(
            state = state,
            viewModel = viewModel,
            isRecording = isRecording,
            onRetryBind = { bindAttempt++ },
            onFinish = {
                viewModel.stopRecording()
                onShowResults()
            }
        )
    }

    if (showLeaveDialog) {
        LeaveDialog(
            onConfirm = {
                showLeaveDialog = false
                viewModel.stopRecording()
                onLeave()
            },
            onDismiss = { showLeaveDialog = false }
        )
    }
}

// ---------------------------------------------------------------------------------------- states

@Composable
private fun LoadingContent() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .safeDrawingPadding(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        CircularProgressIndicator()
        Spacer(Modifier.height(12.dp))
        Text(text = stringResource(R.string.loading), style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun LoadFailedContent(error: Throwable?, onBack: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .safeDrawingPadding()
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = stringResource(R.string.live_load_failed),
            style = MaterialTheme.typography.titleMedium,
            textAlign = TextAlign.Center
        )
        if (error != null) {
            Spacer(Modifier.height(8.dp))
            Text(
                text = stringResource(R.string.error_generic, error.message ?: error.toString()),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.error,
                textAlign = TextAlign.Center
            )
        }
        Spacer(Modifier.height(24.dp))
        Button(onClick = onBack) { Text(stringResource(R.string.action_back)) }
    }
}

@Composable
private fun ReadyContent(
    state: LiveExperimentViewModel.LoadState.Ready,
    viewModel: LiveExperimentViewModel,
    isRecording: Boolean,
    onRetryBind: () -> Unit,
    onFinish: () -> Unit
) {
    // The stored zoom is part of the calibration. The view model applies it while loading, but
    // the pipeline only knows the real zoom range once the camera is bound, so the value is
    // applied again after every successful bind.
    val bindState by viewModel.pipeline.bindState.collectAsStateWithLifecycle()
    LaunchedEffect(bindState, state.config.zoomRatio) {
        if (bindState is CameraPipeline.BindState.Bound) {
            viewModel.pipeline.setZoomRatio(state.config.zoomRatio)
        }
    }

    val isLandscape = LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE
    if (isLandscape) {
        LandscapeContent(state, viewModel, isRecording, onRetryBind, onFinish)
    } else {
        PortraitContent(state, viewModel, isRecording, onRetryBind, onFinish)
    }
}

// --------------------------------------------------------------------------------------- layouts

@Composable
private fun PortraitContent(
    state: LiveExperimentViewModel.LoadState.Ready,
    viewModel: LiveExperimentViewModel,
    isRecording: Boolean,
    onRetryBind: () -> Unit,
    onFinish: () -> Unit
) {
    BoxWithConstraints(
        modifier = Modifier
            .fillMaxSize()
            .safeDrawingPadding()
    ) {
        val cameraMaxHeight = maxHeight * PORTRAIT_CAMERA_MAX_HEIGHT_FRACTION
        Column(modifier = Modifier.fillMaxSize()) {
            CameraPanel(
                viewModel = viewModel,
                roi = state.config.roi,
                ruler = state.config.ruler,
                onRetryBind = onRetryBind,
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(max = cameraMaxHeight),
                maxBoxWidth = PortraitCameraMaxWidth
            )
            TrackingChart(
                model = state.chart,
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(horizontal = 4.dp),
                axisWindow = viewModel.axisWindow(state.chart.latestTimeMs),
                trackerCount = state.config.trackers.size
            )
            WriterErrorText(viewModel)
            StatusRow(viewModel = viewModel, isRecording = isRecording)
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(start = 16.dp, end = 16.dp, bottom = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                StartStopButton(isRecording, viewModel::toggleRecording, Modifier.weight(1f))
                FinishButton(onFinish, Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun LandscapeContent(
    state: LiveExperimentViewModel.LoadState.Ready,
    viewModel: LiveExperimentViewModel,
    isRecording: Boolean,
    onRetryBind: () -> Unit,
    onFinish: () -> Unit
) {
    var panelMode by rememberSaveable { mutableStateOf(PanelMode.CAMERA) }

    Row(
        modifier = Modifier
            .fillMaxSize()
            .safeDrawingPadding()
    ) {
        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxHeight()
        ) {
            TrackingChart(
                model = state.chart,
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(start = 4.dp),
                axisWindow = viewModel.axisWindow(state.chart.latestTimeMs),
                trackerCount = state.config.trackers.size
            )
            WriterErrorText(viewModel)
            StatusRow(viewModel = viewModel, isRecording = isRecording)
        }
        when (panelMode) {
            PanelMode.CAMERA -> CameraPanel(
                viewModel = viewModel,
                roi = state.config.roi,
                ruler = state.config.ruler,
                onRetryBind = onRetryBind,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .padding(8.dp)
            )
            PanelMode.OPTIONS -> OptionsPanel(
                viewModel = viewModel,
                latestTimeMs = state.chart.latestTimeMs,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp)
            )
            PanelMode.PLOT_ONLY -> Unit
        }
        Column(
            modifier = Modifier
                .fillMaxHeight()
                .width(IntrinsicSize.Max)
                .padding(horizontal = 4.dp, vertical = 8.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            PanelIconButton(
                icon = R.drawable.baseline_show_chart_24,
                label = R.string.live_panel_plot,
                isSelected = panelMode == PanelMode.PLOT_ONLY,
                onClick = { panelMode = PanelMode.PLOT_ONLY }
            )
            PanelIconButton(
                icon = R.drawable.baseline_photo_camera_24,
                label = R.string.live_panel_camera,
                isSelected = panelMode == PanelMode.CAMERA,
                onClick = { panelMode = PanelMode.CAMERA }
            )
            PanelIconButton(
                icon = R.drawable.baseline_control_camera_24,
                label = R.string.live_panel_options,
                isSelected = panelMode == PanelMode.OPTIONS,
                onClick = { panelMode = PanelMode.OPTIONS }
            )
            Spacer(Modifier.weight(1f))
            StartStopButton(isRecording, viewModel::toggleRecording, Modifier.fillMaxWidth())
            FinishButton(onFinish, Modifier.fillMaxWidth())
        }
    }
}

// ---------------------------------------------------------------------------------------- camera

@Composable
private fun CameraPanel(
    viewModel: LiveExperimentViewModel,
    roi: Roi?,
    ruler: Ruler?,
    onRetryBind: () -> Unit,
    modifier: Modifier = Modifier,
    maxBoxWidth: Dp = Dp.Unspecified
) {
    val surfaceRequest by viewModel.pipeline.surfaceRequests.collectAsStateWithLifecycle()
    val bindState by viewModel.pipeline.bindState.collectAsStateWithLifecycle()
    val trackedObjects by viewModel.trackedObjects.collectAsStateWithLifecycle()
    val failure = bindState as? CameraPipeline.BindState.Failed

    CameraFrameBox(
        surfaceRequest = if (failure == null) surfaceRequest else null,
        modifier = modifier,
        maxBoxWidth = maxBoxWidth
    ) { mapper ->
        when {
            failure != null -> CameraError(
                error = failure.error,
                onRetry = onRetryBind,
                modifier = Modifier
                    .align(Alignment.Center)
                    .padding(16.dp)
            )
            surfaceRequest == null -> CircularProgressIndicator(Modifier.align(Alignment.Center))
            // The objects are read inside the draw pass only, so a new frame redraws the overlay
            // without recomposing anything.
            else -> TrackingOverlay(mapper, roi, ruler) { trackedObjects }
        }
    }
}

@Composable
private fun CameraError(error: Throwable, onRetry: () -> Unit, modifier: Modifier = Modifier) {
    Column(modifier = modifier, horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = stringResource(R.string.camera_bind_failed, error.message ?: ""),
            color = Color.White,
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center
        )
        Spacer(Modifier.height(8.dp))
        Button(onClick = onRetry) { Text(stringResource(R.string.action_retry)) }
    }
}

@Composable
private fun BoxScope.TrackingOverlay(
    mapper: FrameMapper,
    roi: Roi?,
    ruler: Ruler?,
    objects: () -> List<TrackedObject>
) {
    Canvas(modifier = Modifier.matchParentSize()) {
        if (roi != null) {
            val rect = mapper.toBoxRect(roi.p1, roi.p2)
            drawRect(
                color = RoiGreen,
                topLeft = rect.topLeft,
                size = rect.size,
                style = Stroke(width = 2.dp.toPx())
            )
        }
        if (ruler != null) {
            drawLine(
                color = RulerCyan,
                start = mapper.toBox(ruler.p1),
                end = mapper.toBox(ruler.p2),
                strokeWidth = 1.5.dp.toPx(),
                cap = StrokeCap.Round
            )
        }
        val boxStroke = Stroke(width = 3.dp.toPx())
        for (obj in objects()) {
            val rect = mapper.toBoxRect(Point2D(obj.left, obj.top), Point2D(obj.right, obj.bottom))
            drawRect(
                color = TrackerColors.color(obj.trackerId),
                topLeft = rect.topLeft,
                size = rect.size,
                style = boxStroke
            )
        }
    }
}

// --------------------------------------------------------------------------------------- options

@Composable
private fun OptionsPanel(viewModel: LiveExperimentViewModel, latestTimeMs: Float, modifier: Modifier = Modifier) {
    val offsetLabel = if (viewModel.offsetSlider >= 1f) {
        stringResource(R.string.live_offset_live)
    } else {
        stringResource(R.string.live_offset_seconds, viewModel.offsetMs(latestTimeMs) / 1000L)
    }
    Column(modifier = modifier) {
        Text(
            text = stringResource(R.string.live_time_window, formatDuration(viewModel.timeWindowMs)),
            style = MaterialTheme.typography.titleSmall
        )
        Slider(
            value = viewModel.timeWindowSlider,
            onValueChange = viewModel::onTimeWindowSliderChange,
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(Modifier.height(8.dp))
        Text(
            text = stringResource(R.string.live_offset, offsetLabel),
            style = MaterialTheme.typography.titleSmall
        )
        Slider(
            value = viewModel.offsetSlider,
            onValueChange = viewModel::onOffsetSliderChange,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

// ---------------------------------------------------------------------------------------- status

@Composable
private fun StatusRow(
    viewModel: LiveExperimentViewModel,
    isRecording: Boolean,
    modifier: Modifier = Modifier
) {
    val elapsedMs by viewModel.elapsedMs.collectAsStateWithLifecycle()
    val sampleCount by viewModel.sampleCount.collectAsStateWithLifecycle()

    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        RecordingIndicator(isRecording)
        Text(
            text = stringResource(if (isRecording) R.string.live_status_recording else R.string.live_status_stopped),
            style = MaterialTheme.typography.labelLarge
        )
        Text(
            text = formatElapsed(elapsedMs),
            style = MaterialTheme.typography.bodyMedium.copy(fontFeatureSettings = "tnum")
        )
        Spacer(Modifier.weight(1f))
        Text(
            text = pluralStringResource(R.plurals.live_sample_count, sampleCount.toPluralCount(), sampleCount),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@Composable
private fun RecordingIndicator(isRecording: Boolean) {
    val color = if (isRecording) RecordingRed else MaterialTheme.colorScheme.outlineVariant
    Box(
        modifier = Modifier
            .size(10.dp)
            .background(color, CircleShape)
    )
}

@Composable
private fun WriterErrorText(viewModel: LiveExperimentViewModel) {
    val error by viewModel.writerError.collectAsStateWithLifecycle()
    val current = error ?: return
    Text(
        text = stringResource(R.string.live_writer_error, current.message ?: current.toString()),
        color = MaterialTheme.colorScheme.error,
        style = MaterialTheme.typography.bodySmall,
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
    )
}

// --------------------------------------------------------------------------------------- buttons

@Composable
private fun StartStopButton(isRecording: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    val colors = if (isRecording) {
        ButtonDefaults.buttonColors(containerColor = RecordingRed, contentColor = Color.White)
    } else {
        ButtonDefaults.buttonColors(containerColor = StartGreen, contentColor = Color.White)
    }
    Button(onClick = onClick, modifier = modifier, colors = colors) {
        Text(stringResource(if (isRecording) R.string.live_action_stop else R.string.live_action_start))
    }
}

@Composable
private fun FinishButton(onClick: () -> Unit, modifier: Modifier = Modifier) {
    OutlinedButton(onClick = onClick, modifier = modifier) {
        Text(stringResource(R.string.live_action_finish))
    }
}

@Composable
private fun PanelIconButton(
    @DrawableRes icon: Int,
    @StringRes label: Int,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    val tint = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
    IconButton(
        onClick = onClick,
        modifier = Modifier.semantics { selected = isSelected },
        colors = IconButtonDefaults.iconButtonColors(contentColor = tint)
    ) {
        Icon(painter = painterResource(icon), contentDescription = stringResource(label))
    }
}

// ---------------------------------------------------------------------------------------- dialog

@Composable
private fun LeaveDialog(onConfirm: () -> Unit, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.live_leave_title)) },
        text = { Text(stringResource(R.string.live_leave_message)) },
        confirmButton = {
            TextButton(onClick = onConfirm) { Text(stringResource(R.string.live_leave_confirm)) }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text(stringResource(R.string.action_cancel)) }
        }
    )
}

// ------------------------------------------------------------------------------------ formatting

/** Milliseconds since the experiment start as m:ss, or h:mm:ss from one hour on. */
@Composable
private fun formatElapsed(ms: Long): String {
    val totalSeconds = ms.coerceAtLeast(0L) / 1000L
    val seconds = (totalSeconds % 60L).toInt()
    val minutes = ((totalSeconds / 60L) % 60L).toInt()
    val hours = (totalSeconds / 3600L).toInt()
    return if (hours > 0) {
        stringResource(R.string.live_elapsed_hours, hours, minutes, seconds)
    } else {
        stringResource(R.string.live_elapsed_minutes, minutes, seconds)
    }
}

/** A duration in the most readable unit: seconds below a minute, minutes below an hour, else hours. */
@Composable
private fun formatDuration(ms: Long): String = when {
    ms < 60_000L -> stringResource(R.string.live_duration_seconds, (ms / 1000f).roundToInt().coerceAtLeast(1))
    ms < 3_600_000L -> stringResource(R.string.live_duration_minutes, ms / 60_000f)
    else -> stringResource(R.string.live_duration_hours, ms / 3_600_000f)
}

private fun Long.toPluralCount(): Int = coerceIn(0L, Int.MAX_VALUE.toLong()).toInt()
