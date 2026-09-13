package de.tudarmstadt.physics.trackingplot.ui.experiments_overview

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LifecycleResumeEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import de.tudarmstadt.physics.trackingplot.R
import de.tudarmstadt.physics.trackingplot.model.ExperimentSummary
import java.text.DateFormat
import java.util.Date

/** Bottom padding of the list so that the last entry can be scrolled above the floating action button. */
private val ListBottomPadding = 96.dp

/** Start screen: all stored experiments, newest first, and the entry point to a new one. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ExperimentsOverviewScreen(
    toSetup: () -> Unit,
    toPastExperiment: (experimentId: Long) -> Unit,
    viewModel: ExperimentsOverviewViewModel
) {
    val experiments by viewModel.experiments.collectAsStateWithLifecycle()
    val error by viewModel.error.collectAsStateWithLifecycle()
    val snackbarHostState = remember { SnackbarHostState() }

    // Experiments are created and deleted on other screens: reload whenever this one is shown again.
    LifecycleResumeEffect(Unit) {
        viewModel.refresh()
        onPauseOrDispose { }
    }
    ErrorSnackbar(error = error, hostState = snackbarHostState)

    Scaffold(
        modifier = Modifier
            .fillMaxSize()
            .safeDrawingPadding(),
        topBar = { TopAppBar(title = { Text(stringResource(R.string.app_name)) }) },
        snackbarHost = { SnackbarHost(snackbarHostState) },
        floatingActionButton = { NewExperimentButton(onClick = toSetup) }
    ) { innerPadding ->
        ExperimentList(
            experiments = experiments,
            onOpen = toPastExperiment,
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        )
    }
}

@Composable
private fun ErrorSnackbar(error: Throwable?, hostState: SnackbarHostState) {
    val message = error?.let { stringResource(R.string.error_generic, it.displayMessage()) }
    LaunchedEffect(error) {
        if (message != null) hostState.showSnackbar(message)
    }
}

@Composable
private fun NewExperimentButton(onClick: () -> Unit) {
    ExtendedFloatingActionButton(
        text = { Text(stringResource(R.string.overview_new_experiment)) },
        icon = { Icon(painterResource(R.drawable.baseline_timeline_24), contentDescription = null) },
        onClick = onClick
    )
}

@Composable
private fun ExperimentList(
    experiments: List<ExperimentSummary>?,
    onOpen: (experimentId: Long) -> Unit,
    modifier: Modifier = Modifier
) {
    Box(modifier = modifier, contentAlignment = Alignment.Center) {
        when {
            experiments == null -> CircularProgressIndicator()
            experiments.isEmpty() -> EmptyListHint()
            else -> {
                val dateFormat = remember { DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT) }
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(bottom = ListBottomPadding)
                ) {
                    items(experiments, key = { it.experimentId }) { summary ->
                        ExperimentRow(
                            summary = summary,
                            dateFormat = dateFormat,
                            onClick = { onOpen(summary.experimentId) }
                        )
                        HorizontalDivider()
                    }
                }
            }
        }
    }
}

@Composable
private fun EmptyListHint() {
    Column(
        modifier = Modifier.padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = stringResource(R.string.overview_empty),
            style = MaterialTheme.typography.titleMedium,
            textAlign = TextAlign.Center
        )
        Spacer(Modifier.height(8.dp))
        Text(
            text = stringResource(R.string.overview_empty_hint),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center
        )
    }
}

@Composable
private fun ExperimentRow(summary: ExperimentSummary, dateFormat: DateFormat, onClick: () -> Unit) {
    val description = summary.description.trim()
    val headline = if (description.isEmpty()) {
        stringResource(R.string.experiment_title, summary.experimentId)
    } else {
        stringResource(R.string.experiment_title_described, summary.experimentId, description)
    }
    val samples = pluralStringResource(R.plurals.sample_count, summary.sampleCount.toPluralCount(), summary.sampleCount)
    val trackers = pluralStringResource(R.plurals.tracker_count, summary.trackerCount, summary.trackerCount)
    val supporting = if (summary.createdAt > 0) {
        val date = remember(summary.createdAt, dateFormat) { dateFormat.format(Date(summary.createdAt)) }
        stringResource(R.string.overview_item_supporting, date, samples, trackers)
    } else {
        stringResource(R.string.overview_item_supporting_no_date, samples, trackers)
    }

    ListItem(
        headlineContent = { Text(headline, maxLines = 2, overflow = TextOverflow.Ellipsis) },
        supportingContent = { Text(supporting) },
        modifier = Modifier.clickable(onClick = onClick)
    )
}

private fun Long.toPluralCount(): Int = coerceIn(0L, Int.MAX_VALUE.toLong()).toInt()

private fun Throwable.displayMessage(): String =
    message?.takeIf { it.isNotBlank() } ?: javaClass.simpleName
