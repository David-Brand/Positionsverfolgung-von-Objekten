package de.tudarmstadt.physics.trackingplot.ui.setup.tracker

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.activity.compose.BackHandler
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import de.tudarmstadt.physics.trackingplot.R
import de.tudarmstadt.physics.trackingplot.model.ColorTrackerConfig
import de.tudarmstadt.physics.trackingplot.model.Point2D
import de.tudarmstadt.physics.trackingplot.model.TrackedObject
import de.tudarmstadt.physics.trackingplot.model.TrackingConfig
import de.tudarmstadt.physics.trackingplot.ui.common.FrameMapper
import de.tudarmstadt.physics.trackingplot.ui.common.TrackerColors
import de.tudarmstadt.physics.trackingplot.ui.setup.RectOutline
import de.tudarmstadt.physics.trackingplot.ui.setup.SetupRoiColor
import de.tudarmstadt.physics.trackingplot.ui.setup.SetupScreenScaffold
import de.tudarmstadt.physics.trackingplot.ui.setup.SetupViewModel
import kotlin.math.roundToInt

/** Step 3 of 3: one colour tracker per slot, then start the experiment. */
@Composable
fun TrackerSetupScreen(back: () -> Unit, toLiveExperiment: (experimentId: Long) -> Unit, setupViewModel: SetupViewModel) {
    DisposableEffect(Unit) {
        setupViewModel.setTrackerPreviewEnabled(true)
        onDispose { setupViewModel.setTrackerPreviewEnabled(false) }
    }

    LaunchedEffect(Unit) {
        setupViewModel.events.collect { event ->
            when (event) {
                is SetupViewModel.UiEvent.ToLiveExperiment -> toLiveExperiment(event.experimentId)
            }
        }
    }

    val pickingSlot = setupViewModel.pickingSlot
    val snapshot = setupViewModel.snapshot
    // While the colour picker is open, Back closes it instead of leaving the screen.
    BackHandler(enabled = pickingSlot != null) { setupViewModel.cancelColorPick() }
    if (pickingSlot != null && snapshot != null) {
        ColorPickerOverlay(
            bitmap = snapshot,
            onColorSelected = { color -> setupViewModel.confirmColorPick(color.toArgb()) },
            onCancel = { setupViewModel.cancelColorPick() }
        )
    } else {
        TrackerSetupContent(back = back, setupViewModel = setupViewModel)
    }
}

@Composable
private fun TrackerSetupContent(back: () -> Unit, setupViewModel: SetupViewModel) {
    val trackedObjects by setupViewModel.trackedObjects.collectAsStateWithLifecycle()
    val roi = setupViewModel.roi
    val snapshotError = setupViewModel.snapshotError
    val snackbarHostState = remember { SnackbarHostState() }
    var showStartDialog by remember { mutableStateOf(false) }
    val genericErrorFormat = stringResource(R.string.error_generic)

    LaunchedEffect(snapshotError) {
        val error = snapshotError ?: return@LaunchedEffect
        snackbarHostState.showSnackbar(genericErrorFormat.format(error.message ?: ""))
        setupViewModel.clearSnapshotError()
    }

    Box(modifier = Modifier.fillMaxSize().safeDrawingPadding()) {
        SetupScreenScaffold(
            step = 3,
            title = stringResource(R.string.setup_tracker_title),
            setupViewModel = setupViewModel,
            cameraOverlay = { mapper ->
                roi?.let {
                    val rect = mapper.toBoxRect(Point2D(it.left, it.top), Point2D(it.right, it.bottom))
                    RectOutline(rect = rect, color = SetupRoiColor)
                }
                trackedObjects.forEach { obj -> TrackedObjectOverlay(obj = obj, mapper = mapper) }
            },
            controls = {
                setupViewModel.trackerSlots.forEachIndexed { index, slot ->
                    TrackerSlotCard(
                        index = index,
                        slot = slot,
                        canRemove = setupViewModel.trackerSlots.size > 1,
                        isCapturingSnapshot = setupViewModel.isCapturingSnapshot,
                        onPickColor = { setupViewModel.startColorPick(index) },
                        onToleranceChange = { setupViewModel.setTrackerTolerance(index, it) },
                        onRemove = { setupViewModel.removeTrackerSlot(index) }
                    )
                }

                if (setupViewModel.canAddTracker) {
                    TextButton(onClick = { setupViewModel.addTrackerSlot() }) {
                        Text(stringResource(R.string.setup_add_tracker))
                    }
                }

                StartProblemHint(setupViewModel.startProblem)
            },
            bottomBar = {
                Row(modifier = Modifier.fillMaxWidth()) {
                    OutlinedButton(onClick = back, modifier = Modifier.weight(1f)) {
                        Text(stringResource(R.string.action_back))
                    }
                    Spacer(modifier = Modifier.width(12.dp))
                    Button(
                        onClick = { showStartDialog = true },
                        enabled = setupViewModel.startProblem == null,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text(stringResource(R.string.setup_start_experiment))
                    }
                }
            }
        )
        SnackbarHost(hostState = snackbarHostState, modifier = Modifier.align(Alignment.BottomCenter))
    }

    if (showStartDialog) {
        StartExperimentDialog(setupViewModel = setupViewModel, onDismiss = { showStartDialog = false })
    }
}

@Composable
private fun BoxScope.TrackedObjectOverlay(obj: TrackedObject, mapper: FrameMapper) {
    val rect = mapper.toBoxRect(Point2D(obj.left, obj.top), Point2D(obj.right, obj.bottom))
    val color = TrackerColors.color(obj.trackerId)
    RectOutline(rect = rect, color = color)
    val labelOffsetPx = with(LocalDensity.current) { 20.dp.toPx() }
    Text(
        text = TrackerColors.name(obj.trackerId),
        color = Color.White,
        style = MaterialTheme.typography.labelSmall,
        modifier = Modifier
            .offset {
                IntOffset(rect.left.roundToInt(), (rect.top - labelOffsetPx).roundToInt().coerceAtLeast(0))
            }
            .background(color.copy(alpha = 0.8f))
            .padding(horizontal = 4.dp, vertical = 1.dp)
    )
}

@Composable
private fun TrackerSlotCard(
    index: Int,
    slot: SetupViewModel.TrackerSlot,
    canRemove: Boolean,
    isCapturingSnapshot: Boolean,
    onPickColor: () -> Unit,
    onToleranceChange: (Int) -> Unit,
    onRemove: () -> Unit
) {
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                ColorSwatch(colorArgb = slot.color)
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = TrackerColors.name(index),
                    style = MaterialTheme.typography.titleSmall,
                    modifier = Modifier.weight(1f)
                )
                TextButton(onClick = onPickColor, enabled = !isCapturingSnapshot) {
                    Text(stringResource(R.string.setup_tracker_pick_color))
                }
                if (canRemove) {
                    IconButton(onClick = onRemove, modifier = Modifier.size(32.dp)) {
                        Icon(
                            imageVector = Icons.Filled.Close,
                            contentDescription = stringResource(R.string.setup_tracker_remove),
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
            }
            if (slot.color != null) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = stringResource(R.string.setup_tracker_tolerance, slot.tolerance),
                        style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier.width(84.dp)
                    )
                    Slider(
                        value = slot.tolerance.toFloat(),
                        onValueChange = { onToleranceChange(it.roundToInt()) },
                        valueRange = ColorTrackerConfig.MIN_TOLERANCE.toFloat()..ColorTrackerConfig.MAX_TOLERANCE.toFloat(),
                        modifier = Modifier.weight(1f)
                    )
                }
                if (slot.hsv?.isTrackable == false) {
                    Text(
                        text = stringResource(R.string.setup_tracker_not_trackable),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error
                    )
                }
            }
        }
    }
}

@Composable
private fun ColorSwatch(colorArgb: Int?) {
    if (colorArgb != null) {
        Box(
            modifier = Modifier
                .size(20.dp)
                .clip(CircleShape)
                .background(Color(colorArgb))
        )
    } else {
        Canvas(modifier = Modifier.size(20.dp)) {
            drawCircle(
                color = Color.Gray,
                radius = size.minDimension / 2f - 1.5.dp.toPx(),
                style = Stroke(width = 1.5.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(4f, 4f)))
            )
        }
    }
}

@Composable
private fun StartProblemHint(problem: SetupViewModel.StartProblem?) {
    if (problem == null) return
    val text = when (problem) {
        SetupViewModel.StartProblem.NO_TRACKER -> stringResource(R.string.setup_start_problem_no_tracker)
        SetupViewModel.StartProblem.TRACKER_WITHOUT_COLOR -> stringResource(R.string.setup_start_problem_tracker_without_color)
        SetupViewModel.StartProblem.INVALID_SAMPLING_RATE -> stringResource(R.string.setup_start_problem_invalid_sampling_rate)
    }
    Text(text = text, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
}

@Composable
private fun StartExperimentDialog(setupViewModel: SetupViewModel, onDismiss: () -> Unit) {
    val startError = setupViewModel.startError
    val genericErrorFormat = stringResource(R.string.error_generic)

    fun dismiss() {
        if (!setupViewModel.isStarting) {
            setupViewModel.clearStartError()
            onDismiss()
        }
    }

    AlertDialog(
        onDismissRequest = ::dismiss,
        title = { Text(stringResource(R.string.setup_start_dialog_title)) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(
                    value = setupViewModel.samplingRateText,
                    onValueChange = { setupViewModel.samplingRateText = it },
                    label = { Text(stringResource(R.string.setup_sampling_rate_label)) },
                    supportingText = {
                        Text(
                            stringResource(
                                R.string.setup_sampling_rate_supporting,
                                TrackingConfig.MIN_SAMPLING_RATE,
                                TrackingConfig.MAX_SAMPLING_RATE
                            )
                        )
                    },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
                OutlinedTextField(
                    value = setupViewModel.description,
                    onValueChange = { setupViewModel.description = it },
                    label = { Text(stringResource(R.string.setup_description_label)) },
                    modifier = Modifier.fillMaxWidth()
                )
                if (startError != null) {
                    Text(
                        text = genericErrorFormat.format(startError.message ?: ""),
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        },
        confirmButton = {
            Button(
                onClick = { setupViewModel.startExperiment() },
                enabled = setupViewModel.startProblem == null && !setupViewModel.isStarting
            ) {
                Text(stringResource(R.string.setup_start_experiment))
            }
        },
        dismissButton = {
            TextButton(onClick = ::dismiss, enabled = !setupViewModel.isStarting) {
                Text(stringResource(R.string.action_cancel))
            }
        }
    )
}
