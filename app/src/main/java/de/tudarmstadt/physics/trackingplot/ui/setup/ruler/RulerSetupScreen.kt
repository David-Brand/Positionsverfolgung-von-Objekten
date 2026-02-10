package de.tudarmstadt.physics.trackingplot.ui.setup.ruler

import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceRequest
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import de.tudarmstadt.physics.trackingplot.ui.CameraWithMarkersRuler
import de.tudarmstadt.physics.trackingplot.ui.setup.SetupViewModel

@Composable
fun RulerSetupScreen(
    setupViewModel: SetupViewModel

) {
    //TODO setup CameraXViewfinder with surfaceRequest
    // store points in viewModel
    // draw points on canvas above Viewfinder
    /*
    Canvas(
        modifier = Modifier.fillMaxSize()
    ) {
        setupViewModel.uiPoints.forEach {
        }
    }

     */

    // Hier halten wir das Request, sobald die Kamera bereit ist
    var surfaceRequest by remember { mutableStateOf<SurfaceRequest?>(null) }
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    // Kamera-Initialisierung
    LaunchedEffect(Unit) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Preview konfigurieren
            val preview = Preview.Builder().build()
            preview.setSurfaceProvider { request ->
                surfaceRequest = request
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
                Log.e("Camera", "Binding failed", e)
            }
        }, ContextCompat.getMainExecutor(context)) // Führt den Code auf dem Main-Thread aus
    }

    Box(modifier = Modifier.fillMaxSize()) {

        // 1. Die Kamera-Komponente aufrufen
        CameraWithMarkersRuler(
            surfaceRequest = surfaceRequest,
            viewModel = setupViewModel,
            onDone = { imageCoords ->
                // Wenn der User fertig ist, rufen wir die Speicher-Logik im VM auf
                setupViewModel.storeExperimentSetupAndStart()
            }
        )

        // 2. Zusätzlicher Canvas (falls du außerhalb noch was zeichnen willst)
        Canvas(modifier = Modifier.fillMaxSize()) {
            setupViewModel.uiPoints.forEach { point ->
                // Hier könntest du z.B. kleine Beschriftungen ("Punkt A") zeichnen
            }
        }
    }
}