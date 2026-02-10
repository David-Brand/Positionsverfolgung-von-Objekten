package de.tudarmstadt.physics.trackingplot.ui

import androidx.camera.compose.CameraXViewfinder
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceRequest
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.lifecycle.awaitInstance
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.LocalLifecycleOwner
import kotlinx.coroutines.flow.MutableStateFlow

@Composable
fun NewCameraPreview() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val selector = CameraSelector.DEFAULT_BACK_CAMERA

    val surfaceRequests = remember { MutableStateFlow<SurfaceRequest?>(null) }
    val surfaceRequest by surfaceRequests.collectAsState(initial = null)

    LaunchedEffect(Unit) {
        val provider = ProcessCameraProvider.awaitInstance(context)
        val preview = Preview.Builder().build().apply {
            setSurfaceProvider { req -> surfaceRequests.value = req }
        }
        provider.unbindAll()
        provider.bindToLifecycle(lifecycleOwner, selector, preview)
    }

    surfaceRequest?.let {
        CameraWithMarkers(
            surfaceRequest = it,
            onDone = {
                println()
            }
        )
//        CameraXViewfinder(
//            surfaceRequest = it,
//            modifier = Modifier.fillMaxSize()
//        )
    }
}