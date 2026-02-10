package de.tudarmstadt.physics.trackingplot.ui.setup.ruler

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import de.tudarmstadt.physics.trackingplot.ui.setup.SetupViewModel

@Composable
fun RulerSetupScreen(
    setupViewModel: SetupViewModel
) {
    //TODO setup CameraXViewfinder with surfaceRequest
    // store points in viewModel
    // draw points on canvas above Viewfinder
    Canvas(
        modifier = Modifier.fillMaxSize()
    ) {
        setupViewModel.uiPoints.forEach {
        }
    }
}