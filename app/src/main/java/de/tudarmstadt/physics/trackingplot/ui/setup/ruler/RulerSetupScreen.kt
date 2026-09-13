package de.tudarmstadt.physics.trackingplot.ui.setup.ruler

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuAnchorType
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import de.tudarmstadt.physics.trackingplot.R
import de.tudarmstadt.physics.trackingplot.model.DistanceUnit
import de.tudarmstadt.physics.trackingplot.ui.setup.DragHandles
import de.tudarmstadt.physics.trackingplot.ui.setup.LineOverlay
import de.tudarmstadt.physics.trackingplot.ui.setup.SetupRulerColor
import de.tudarmstadt.physics.trackingplot.ui.setup.SetupScreenScaffold
import de.tudarmstadt.physics.trackingplot.ui.setup.SetupViewModel
import kotlin.math.roundToInt

/** Step 1 of 3: an optional real-world reference length used to convert positions to real units. */
@Composable
fun RulerSetupScreen(toNextStep: () -> Unit, setupViewModel: SetupViewModel) {
    val state = setupViewModel.rulerState
    SetupScreenScaffold(
        step = 1,
        title = stringResource(R.string.setup_ruler_title),
        setupViewModel = setupViewModel,
        cameraOverlay = { mapper ->
            val points = setupViewModel.rulerPoints
            if (points.size == 2) {
                LineOverlay(from = mapper.toBox(points[0]), to = mapper.toBox(points[1]), color = SetupRulerColor)
            }
            DragHandles(
                onZoom = setupViewModel::zoomBy,
                points = points,
                mapper = mapper,
                onAddPoint = setupViewModel::addRulerPoint,
                onMovePoint = setupViewModel::moveRulerPoint
            )
        },
        controls = {
            Row(modifier = Modifier.fillMaxWidth()) {
                OutlinedTextField(
                    value = setupViewModel.distanceText,
                    onValueChange = { setupViewModel.distanceText = it },
                    label = { Text(stringResource(R.string.setup_ruler_distance_label)) },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                    singleLine = true,
                    modifier = Modifier.weight(1f)
                )
                DistanceUnitDropdown(
                    selected = setupViewModel.distanceUnit,
                    onSelected = { setupViewModel.distanceUnit = it },
                    modifier = Modifier.width(110.dp)
                )
            }

            Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = rulerStateMessage(state),
                    style = MaterialTheme.typography.bodySmall,
                    color = if (state is SetupViewModel.RulerState.Incomplete) {
                        MaterialTheme.colorScheme.error
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f)
                )
                TextButton(onClick = { setupViewModel.clearRulerPoints() }) {
                    Text(stringResource(R.string.action_reset_points))
                }
            }
        },
        bottomBar = {
            if (state is SetupViewModel.RulerState.Valid) {
                Button(onClick = toNextStep, modifier = Modifier.fillMaxWidth()) {
                    Text(stringResource(R.string.action_next))
                }
            } else {
                OutlinedButton(onClick = toNextStep, modifier = Modifier.fillMaxWidth()) {
                    Text(stringResource(R.string.setup_ruler_skip))
                }
            }
        }
    )
}

@Composable
private fun rulerStateMessage(state: SetupViewModel.RulerState): String = when (state) {
    is SetupViewModel.RulerState.NotSet -> stringResource(R.string.setup_ruler_hint_optional)
    is SetupViewModel.RulerState.Incomplete -> when (state.problem) {
        SetupViewModel.RulerProblem.NEED_POINTS -> stringResource(R.string.setup_ruler_need_points)
        SetupViewModel.RulerProblem.NEED_DISTANCE -> stringResource(R.string.setup_ruler_need_distance)
        SetupViewModel.RulerProblem.INVALID_DISTANCE -> stringResource(R.string.setup_ruler_invalid_distance)
        SetupViewModel.RulerProblem.DISTANCE_NOT_POSITIVE -> stringResource(R.string.setup_ruler_distance_not_positive)
        SetupViewModel.RulerProblem.POINTS_TOO_CLOSE -> stringResource(R.string.setup_ruler_points_too_close)
    }
    is SetupViewModel.RulerState.Valid -> stringResource(
        R.string.setup_ruler_reference_set,
        formatDistance(state.ruler.realDistance),
        state.ruler.unit.symbol
    )
}

private fun formatDistance(value: Float): String {
    val rounded = (value * 100f).roundToInt() / 100f
    return if (rounded == rounded.toLong().toFloat()) rounded.toLong().toString() else rounded.toString()
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun DistanceUnitDropdown(selected: DistanceUnit, onSelected: (DistanceUnit) -> Unit, modifier: Modifier = Modifier) {
    var expanded by remember { mutableStateOf(false) }
    ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = it }, modifier = modifier) {
        OutlinedTextField(
            value = selected.symbol,
            onValueChange = {},
            readOnly = true,
            label = { Text(stringResource(R.string.setup_ruler_unit_label)) },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier
                .menuAnchor(ExposedDropdownMenuAnchorType.PrimaryNotEditable)
                .fillMaxWidth()
        )
        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false }
        ) {
            DistanceUnit.entries.forEach { unit ->
                DropdownMenuItem(
                    text = { Text(unit.symbol) },
                    onClick = {
                        onSelected(unit)
                        expanded = false
                    }
                )
            }
        }
    }
}
