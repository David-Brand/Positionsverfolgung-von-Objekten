package de.tudarmstadt.physics.trackingplot.ui.ruler

import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner

@Composable
fun CameraWithOverlay(
    modifier: Modifier = Modifier,
    onPreviewSizeChanged: (Size) -> Unit = {},
    overlayContent: @Composable (Size) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember { PreviewView(context) }

    AndroidView(
        factory = { previewView },
        modifier = modifier
            .fillMaxSize()
            .onSizeChanged { size ->
                onPreviewSizeChanged(Size(size.width.toFloat(), size.height.toFloat()))
            }
    )

//    Box(modifier = Modifier.matchParentSize()) {
    Box(modifier = Modifier.fillMaxSize()) {
        // Camera is behind
        overlayContent(Size(previewView.width.toFloat(), previewView.height.toFloat()))
    }

    LaunchedEffect(Unit) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview
                )
            } catch (e: Exception) {
                Log.e("Camera", "Bind failed", e)
            }
        }, ContextCompat.getMainExecutor(context))
    }
}