package de.tudarmstadt.physics.trackingplot.ui.live_experiment

import androidx.activity.compose.BackHandler
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import de.tudarmstadt.physics.trackingplot.R

@Composable
fun LiveExperimentScreen(
    onAbortExperiment: () -> Unit,
    viewModel: LiveExperimentViewModel
) {
    var showAbortDialog by remember { mutableStateOf(false) }

    BackHandler(enabled = true) {
        showAbortDialog = true
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