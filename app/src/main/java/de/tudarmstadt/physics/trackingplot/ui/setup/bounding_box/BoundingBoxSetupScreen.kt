package de.tudarmstadt.physics.trackingplot.ui.setup.bounding_box

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import de.tudarmstadt.physics.trackingplot.R
import de.tudarmstadt.physics.trackingplot.ui.setup.DragHandles
import de.tudarmstadt.physics.trackingplot.ui.setup.RectOutline
import de.tudarmstadt.physics.trackingplot.ui.setup.SetupRoiColor
import de.tudarmstadt.physics.trackingplot.ui.setup.SetupScreenScaffold
import de.tudarmstadt.physics.trackingplot.ui.setup.SetupViewModel

/** Step 2 of 3: an optional region of interest that limits where trackers are searched for. */
@Composable
fun BoundingBoxSetupScreen(back: () -> Unit, toNextStep: () -> Unit, setupViewModel: SetupViewModel) {
    SetupScreenScaffold(
        step = 2,
        title = stringResource(R.string.setup_roi_title),
        setupViewModel = setupViewModel,
        cameraOverlay = { mapper ->
            val points = setupViewModel.roiPoints
            if (points.size == 2) {
                val rect = mapper.toBoxRect(points[0], points[1])
                val color = if (setupViewModel.useRoi) SetupRoiColor else SetupRoiColor.copy(alpha = 0.4f)
                RectOutline(rect = rect, color = color)
            }
            DragHandles(
                onZoom = setupViewModel::zoomBy,
                points = points,
                mapper = mapper,
                onAddPoint = setupViewModel::addRoiPoint,
                onMovePoint = setupViewModel::moveRoiPoint
            )
        },
        controls = {
            Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Checkbox(
                    checked = setupViewModel.useRoi,
                    onCheckedChange = { setupViewModel.useRoi = it },
                    enabled = setupViewModel.roiPoints.size == 2
                )
                Text(
                    text = stringResource(R.string.setup_roi_use_checkbox),
                    modifier = Modifier.weight(1f)
                )
                TextButton(onClick = { setupViewModel.clearRoiPoints() }) {
                    Text(stringResource(R.string.action_reset_points))
                }
            }

            if (setupViewModel.roiTooSmall) {
                Text(
                    text = stringResource(R.string.setup_roi_too_small),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error
                )
            }
        },
        bottomBar = {
            Row(modifier = Modifier.fillMaxWidth()) {
                OutlinedButton(onClick = back, modifier = Modifier.weight(1f)) {
                    Text(stringResource(R.string.action_back))
                }
                Spacer(modifier = Modifier.width(12.dp))
                Button(onClick = toNextStep, modifier = Modifier.weight(1f)) {
                    Text(
                        text = if (setupViewModel.roi != null) {
                            stringResource(R.string.action_next)
                        } else {
                            stringResource(R.string.action_skip)
                        }
                    )
                }
            }
        }
    )
}
