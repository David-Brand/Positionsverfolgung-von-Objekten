package de.tudarmstadt.physics.trackingplot.ui.past_experiment

import android.content.res.Configuration
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import de.tudarmstadt.physics.trackingplot.R
import de.tudarmstadt.physics.trackingplot.model.ExperimentSummary
import de.tudarmstadt.physics.trackingplot.model.TrackingConfig
import de.tudarmstadt.physics.trackingplot.ui.common.ChartModel
import de.tudarmstadt.physics.trackingplot.ui.common.TrackingChart
import de.tudarmstadt.physics.trackingplot.ui.past_experiment.PastExperimentViewModel.DeleteState
import de.tudarmstadt.physics.trackingplot.ui.past_experiment.PastExperimentViewModel.ExportState
import de.tudarmstadt.physics.trackingplot.ui.past_experiment.PastExperimentViewModel.State
import kotlinx.coroutines.launch
import java.text.DateFormat
import java.util.Date

private const val CSV_MIME_TYPE = "text/csv"

/** Results of a recorded experiment: summary, plot of all samples, CSV export and deletion. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PastExperimentScreen(viewModel: PastExperimentViewModel, onBack: () -> Unit) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val exportState by viewModel.exportState.collectAsStateWithLifecycle()
    val deleteState by viewModel.deleteState.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val snackbarHostState = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()
    var showDeleteDialog by rememberSaveable { mutableStateOf(false) }

    val exportLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument(CSV_MIME_TYPE)) { uri ->
        uri?.let { viewModel.exportTo(it, context.contentResolver) }
    }
    val showMessage: (String) -> Unit = { text -> scope.launch { snackbarHostState.showSnackbar(text) } }

    ExportStateEffect(state = exportState, showMessage = showMessage, onShown = viewModel::dismissExportState)
    DeleteStateEffect(
        state = deleteState,
        showMessage = showMessage,
        onDeleted = onBack,
        onFailureShown = viewModel::dismissDeleteState
    )

    val actions = ExperimentActions(
        onExport = { exportLauncher.launch(viewModel.csvFileName) },
        onDelete = { showDeleteDialog = true },
        exporting = exportState is ExportState.Exporting,
        // Done: the screen is about to be left; keep the buttons disabled until then.
        deleting = deleteState is DeleteState.Deleting || deleteState is DeleteState.Done
    )

    Scaffold(
        modifier = Modifier
            .fillMaxSize()
            .safeDrawingPadding(),
        topBar = { PastExperimentTopBar(experimentId = viewModel.experimentId, onBack = onBack) },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            when (val current = state) {
                State.Loading -> CircularProgressIndicator(Modifier.align(Alignment.Center))
                is State.Failed -> LoadFailedContent(
                    error = current.error,
                    onRetry = viewModel::reload,
                    onBack = onBack,
                    modifier = Modifier.fillMaxSize()
                )
                is State.Ready -> ReadyContent(
                    summary = current.summary,
                    config = current.config,
                    chart = current.chart,
                    actions = actions,
                    modifier = Modifier.fillMaxSize()
                )
            }
        }
    }

    if (showDeleteDialog) {
        DeleteConfirmationDialog(
            experimentId = viewModel.experimentId,
            onConfirm = {
                showDeleteDialog = false
                viewModel.delete()
            },
            onDismiss = { showDeleteDialog = false }
        )
    }
}

/** What the Ready content can trigger, and whether one of it is currently running. */
private class ExperimentActions(
    val onExport: () -> Unit,
    val onDelete: () -> Unit,
    val exporting: Boolean,
    val deleting: Boolean
) {
    val enabled: Boolean get() = !exporting && !deleting
}

/** Shows the outcome of an export once and marks it as seen in the ViewModel. */
@Composable
private fun ExportStateEffect(state: ExportState, showMessage: (String) -> Unit, onShown: () -> Unit) {
    val message = when (state) {
        ExportState.Done -> stringResource(R.string.past_export_done)
        is ExportState.Failed -> stringResource(R.string.past_export_failed, state.error.displayMessage())
        ExportState.Idle, ExportState.Exporting -> null
    }
    LaunchedEffect(state) {
        if (message != null) {
            showMessage(message)
            onShown()
        }
    }
}

/** Leaves the screen once the experiment is gone; reports a failed deletion once. */
@Composable
private fun DeleteStateEffect(
    state: DeleteState,
    showMessage: (String) -> Unit,
    onDeleted: () -> Unit,
    onFailureShown: () -> Unit
) {
    val failure = (state as? DeleteState.Failed)?.let {
        stringResource(R.string.past_delete_failed, it.error.displayMessage())
    }
    LaunchedEffect(state) {
        when {
            state is DeleteState.Done -> onDeleted()
            failure != null -> {
                showMessage(failure)
                onFailureShown()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PastExperimentTopBar(experimentId: Long, onBack: () -> Unit) {
    TopAppBar(
        title = { Text(stringResource(R.string.experiment_title, experimentId)) },
        navigationIcon = {
            IconButton(onClick = onBack) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = stringResource(R.string.action_back)
                )
            }
        }
    )
}

@Composable
private fun LoadFailedContent(
    error: Throwable?,
    onRetry: () -> Unit,
    onBack: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier.padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = if (error == null) {
                stringResource(R.string.past_not_found)
            } else {
                stringResource(R.string.error_generic, error.displayMessage())
            },
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.error,
            textAlign = TextAlign.Center
        )
        Spacer(Modifier.height(24.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(onClick = onBack) { Text(stringResource(R.string.action_back)) }
            if (error != null) {
                Button(onClick = onRetry) { Text(stringResource(R.string.action_retry)) }
            }
        }
    }
}

@Composable
private fun ReadyContent(
    summary: ExperimentSummary,
    config: TrackingConfig,
    chart: ChartModel,
    actions: ExperimentActions,
    modifier: Modifier = Modifier
) {
    val landscape = LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE
    if (landscape) {
        Row(modifier = modifier) {
            Column(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
            ) {
                ExperimentHeader(summary, config, Modifier.fillMaxWidth())
                ExperimentPlot(summary, config, chart, Modifier.fillMaxWidth().weight(1f))
            }
            Column(
                modifier = Modifier
                    .width(IntrinsicSize.Max)
                    .fillMaxHeight()
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                ExportButton(actions, Modifier.fillMaxWidth())
                DeleteButton(actions, Modifier.fillMaxWidth())
            }
        }
    } else {
        Column(modifier = modifier) {
            ExperimentHeader(summary, config, Modifier.fillMaxWidth())
            ExperimentPlot(summary, config, chart, Modifier.fillMaxWidth().weight(1f))
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                ExportButton(actions, Modifier.weight(1f))
                DeleteButton(actions, Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun ExperimentHeader(summary: ExperimentSummary, config: TrackingConfig, modifier: Modifier = Modifier) {
    val description = summary.description.trim()
    Column(
        modifier = modifier.padding(horizontal = 16.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        if (description.isNotEmpty()) {
            Text(text = description, style = MaterialTheme.typography.titleMedium)
        }
        if (summary.createdAt > 0) {
            val createdAt = remember(summary.createdAt) {
                DateFormat.getDateTimeInstance().format(Date(summary.createdAt))
            }
            Text(
                text = stringResource(R.string.past_recorded_at, createdAt),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        Text(
            text = statsText(summary, config),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun statsText(summary: ExperimentSummary, config: TrackingConfig): String {
    val samples = pluralStringResource(R.plurals.sample_count, summary.sampleCount.toPluralCount(), summary.sampleCount)
    val trackers = pluralStringResource(R.plurals.tracker_count, config.trackers.size, config.trackers.size)
    val rate = if (config.samplingRate > 0) {
        stringResource(R.string.past_rate_per_second, config.samplingRate)
    } else {
        stringResource(R.string.past_rate_every_frame)
    }
    return stringResource(R.string.past_stats, samples, trackers, rate)
}

@Composable
private fun ExperimentPlot(
    summary: ExperimentSummary,
    config: TrackingConfig,
    chart: ChartModel,
    modifier: Modifier = Modifier
) {
    if (summary.sampleCount == 0L) {
        Box(modifier = modifier.padding(32.dp), contentAlignment = Alignment.Center) {
            Text(
                text = stringResource(R.string.past_no_samples),
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center
            )
        }
    } else {
        TrackingChart(
            model = chart,
            modifier = modifier.padding(horizontal = 8.dp),
            trackerCount = config.trackers.size
        )
    }
}

@Composable
private fun ExportButton(actions: ExperimentActions, modifier: Modifier = Modifier) {
    Button(onClick = actions.onExport, enabled = actions.enabled, modifier = modifier) {
        if (actions.exporting) {
            CircularProgressIndicator(
                modifier = Modifier.size(18.dp),
                color = LocalContentColor.current,
                strokeWidth = 2.dp
            )
            Spacer(Modifier.width(8.dp))
        }
        Text(stringResource(R.string.past_export_csv))
    }
}

@Composable
private fun DeleteButton(actions: ExperimentActions, modifier: Modifier = Modifier) {
    OutlinedButton(
        onClick = actions.onDelete,
        enabled = actions.enabled,
        modifier = modifier,
        colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.error)
    ) {
        Text(stringResource(R.string.past_delete))
    }
}

@Composable
private fun DeleteConfirmationDialog(experimentId: Long, onConfirm: () -> Unit, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.past_delete_title)) },
        text = { Text(stringResource(R.string.past_delete_message, experimentId)) },
        confirmButton = {
            TextButton(onClick = onConfirm) {
                Text(stringResource(R.string.past_delete), color = MaterialTheme.colorScheme.error)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text(stringResource(R.string.action_cancel)) }
        }
    )
}

private fun Long.toPluralCount(): Int = coerceIn(0L, Int.MAX_VALUE.toLong()).toInt()

private fun Throwable.displayMessage(): String =
    message?.takeIf { it.isNotBlank() } ?: javaClass.simpleName
