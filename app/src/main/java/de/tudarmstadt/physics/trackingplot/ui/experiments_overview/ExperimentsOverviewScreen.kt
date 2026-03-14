package de.tudarmstadt.physics.trackingplot.ui.experiments_overview

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.ListItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import de.tudarmstadt.physics.trackingplot.R

@Composable
fun ExperimentsOverviewScreen(
    toSetup: () -> Unit,
    toPastExperiment: (experimentId: Long) -> Unit,
    viewModel: ExperimentsOverviewViewModel
) {
    Scaffold(
        floatingActionButton = {
            ExtendedFloatingActionButton(
                text = { Text("New Experiment") },
                icon = { Icon(
                    painter = painterResource(id = R.drawable.baseline_timeline_24),
                    contentDescription = "New Experiment"
                ) },
                onClick = toSetup
            )
        }
    ) { paddingValues ->
        LaunchedEffect(Unit) {
            viewModel.loadExperiments()
        }
        val experiments = viewModel.experiments
        if (experiments.isEmpty()) {
            Box(
                modifier = Modifier
                    .padding(paddingValues)
                    .fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Text("no past experiments")
            }
        } else {
            LazyColumn(
                modifier = Modifier
                    .padding(paddingValues)
                    .fillMaxSize()
            ) {
                items(experiments) { experiment ->
                    ListItem(
                        headlineContent = {
                            Text("Experiment ${experiment.experimentId}")
                        },
                        supportingContent = {
                            Text(experiment.description)
                        },
                        modifier = Modifier
                            .clickable {
                                // TODO ID
                                val experimentId = experiment.experimentId
                                toPastExperiment(experimentId)
                            }
                    )
                    HorizontalDivider()
                }
            }
        }
    }
}