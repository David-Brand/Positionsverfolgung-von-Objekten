package de.tudarmstadt.physics.trackingplot.ui.common

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LifecycleResumeEffect
import de.tudarmstadt.physics.trackingplot.R

/**
 * Shows [content] only while the camera permission is granted. Otherwise the permission is
 * requested (once automatically) and an explanation with a retry / settings button is shown.
 */
@Composable
fun CameraPermissionGate(content: @Composable () -> Unit) {
    val context = LocalContext.current
    var granted by remember { mutableStateOf(hasCameraPermission(context)) }
    var askedAutomatically by rememberSaveable { mutableStateOf(false) }
    var deniedPermanently by remember { mutableStateOf(false) }

    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        granted = isGranted
        if (!isGranted) {
            val activity = context.findActivity()
            deniedPermanently = activity != null &&
                !ActivityCompat.shouldShowRequestPermissionRationale(activity, Manifest.permission.CAMERA)
        }
    }

    // The user may grant the permission in the system settings and come back.
    LifecycleResumeEffect(Unit) {
        granted = hasCameraPermission(context)
        onPauseOrDispose { }
    }

    LaunchedEffect(granted, askedAutomatically) {
        if (!granted && !askedAutomatically) {
            askedAutomatically = true
            launcher.launch(Manifest.permission.CAMERA)
        }
    }

    if (granted) {
        content()
    } else {
        PermissionExplanation(
            deniedPermanently = deniedPermanently,
            onRequest = { launcher.launch(Manifest.permission.CAMERA) },
            onOpenSettings = {
                val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null))
                context.startActivity(intent)
            }
        )
    }
}

@Composable
private fun PermissionExplanation(
    deniedPermanently: Boolean,
    onRequest: () -> Unit,
    onOpenSettings: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .safeDrawingPadding()
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = stringResource(R.string.camera_permission_title),
            style = MaterialTheme.typography.titleLarge,
            textAlign = TextAlign.Center
        )
        Spacer(Modifier.height(12.dp))
        Text(
            text = stringResource(R.string.camera_permission_message),
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center
        )
        Spacer(Modifier.height(24.dp))
        if (deniedPermanently) {
            Button(onClick = onOpenSettings) { Text(stringResource(R.string.camera_permission_open_settings)) }
        } else {
            Button(onClick = onRequest) { Text(stringResource(R.string.camera_permission_grant)) }
            Spacer(Modifier.height(8.dp))
            OutlinedButton(onClick = onOpenSettings) { Text(stringResource(R.string.camera_permission_open_settings)) }
        }
    }
}

fun hasCameraPermission(context: android.content.Context): Boolean =
    ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

private fun android.content.Context.findActivity(): Activity? {
    var current: android.content.Context = this
    while (current is android.content.ContextWrapper) {
        if (current is Activity) return current
        current = current.baseContext
    }
    return null
}
