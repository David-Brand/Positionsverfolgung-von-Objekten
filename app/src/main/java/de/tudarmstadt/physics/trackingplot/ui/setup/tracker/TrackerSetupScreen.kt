package de.tudarmstadt.physics.trackingplot.ui.setup.tracker

import android.graphics.Paint
import androidx.camera.compose.CameraXViewfinder
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.lifecycle.awaitInstance
import androidx.camera.viewfinder.compose.MutableCoordinateTransformer
import androidx.camera.viewfinder.core.ImplementationMode
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import de.tudarmstadt.physics.trackingplot.ui.ObserveAsEvents
import de.tudarmstadt.physics.trackingplot.ui.setup.SetupViewModel

@Composable
fun TrackerSetupScreen(
    back: () -> Unit,
    toLiveExperiment: (experimentId: Long) -> Unit,
    setupViewModel: SetupViewModel
) {
    var showStartDialog by remember { mutableStateOf(false) }

    ObserveAsEvents(flow = setupViewModel.eventsChannelFlow) { event ->
        when (event) {
            is SetupViewModel.UiEvent.ToLiveExperiment -> {
                toLiveExperiment(event.experimentId)
            }
        }
    }

    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val selector = CameraSelector.DEFAULT_BACK_CAMERA

    LaunchedEffect(Unit) {
        val provider = ProcessCameraProvider.awaitInstance(context)

        val imageCapture = ImageCapture.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    .build()
            ).build()
        setupViewModel.imageCapture = imageCapture
        val useCaseGroup = setupViewModel.createUseCaseGroup(
            imageCapture = imageCapture,
            imageAnalysis = setupViewModel.createImageAnalysis()
        )

        provider.unbindAll()
        provider.bindToLifecycle(lifecycleOwner, selector, useCaseGroup)
    }

    val surfaceRequest by setupViewModel.surfaceRequests.collectAsStateWithLifecycle()


    var tracker1Color by remember { mutableStateOf<Color?>(null) }
    var tracker2Color by remember { mutableStateOf<Color?>(null) }
    var tracker3Color by remember { mutableStateOf<Color?>(null) }
    var activeTrackerIndex by remember { mutableStateOf(-1) }

    surfaceRequest?.let {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .safeDrawingPadding()
        ) {
            val bitmap = setupViewModel.frozenBitmap
            val isPicking = setupViewModel.isPickingColor
            if (isPicking && bitmap != null) {
                ColorPickerOverlay(
                    bitmap = bitmap,
                    onColorChanged = {},
                    onColorSelected = { color ->
                        setupViewModel.isPickingColor = false
                        when (activeTrackerIndex) {
                            1 -> {
                                setupViewModel.trackerColorSelected(
                                    0,
                                    color,
                                    45
                                )
                                tracker1Color = color
                            }
                            2 -> {
                                setupViewModel.trackerColorSelected(
                                    1,
                                    color,
                                    45
                                )
                                tracker2Color = color
                            }
                            3 -> {
                                setupViewModel.trackerColorSelected(
                                    2,
                                    color,
                                    45
                                )
                                tracker3Color = color
                            }
                        }
                    },
                    onCancel = { setupViewModel.isPickingColor = false }
                )
            } else {
//                Scaffold(
//                    floatingActionButton = {
//                        ExtendedFloatingActionButton(
//                            text = { Text("Start Experiment") },
//                            icon = { Icon(
//                                painter = painterResource(id = R.drawable.baseline_timeline_24),
//                                contentDescription = "Start Experiment"
//                            ) },
//                            onClick = {
//                                setupViewModel.storeExperimentSetupAndStart()
//                            }
//                        )
//                    }
//                ) { paddingValues ->
//                }
                BoxWithConstraints(
                    modifier = Modifier
//                        .padding(paddingValues)
                        .fillMaxWidth()
                        .weight(1f),
                    contentAlignment = Alignment.Center
                ) {
                    val aspectRatio = 3 to 4
                    val maxAspectRatio: Float = maxWidth / maxHeight
                    val aspectRatioFloat: Float = aspectRatio.first.toFloat() / aspectRatio.second
                    val shouldUseMaxWidth = maxAspectRatio <= aspectRatioFloat
                    val width = if (shouldUseMaxWidth) maxWidth else maxHeight * aspectRatioFloat
                    val height = if (!shouldUseMaxWidth) maxHeight else maxWidth / aspectRatioFloat

                    Box(
                        modifier = Modifier
                            .width(width)
                            .height(height)
                    ) {
                        val implementationMode = ImplementationMode.EXTERNAL

                        val coordinateTransformer = remember { MutableCoordinateTransformer() }
                        var draggedIndex by remember { mutableStateOf(-1) }
                        CameraXViewfinder(
                            modifier = Modifier
                                .fillMaxSize(),
                            surfaceRequest = it,
                            implementationMode = implementationMode,
                            coordinateTransformer = coordinateTransformer,
                            contentScale = ContentScale.Fit
                        )
                        val boxes by setupViewModel.boxes.collectAsStateWithLifecycle()
                        val boundingBoxUiPoints = setupViewModel.boundingUiPoints
                        val uiPoints = setupViewModel.boundingUiPoints
                        val useBoundingBox = setupViewModel.useBoundingBox
                        Canvas(
                            modifier = Modifier.matchParentSize()
                        ) {
                            val canvasWidth = size.width
                            val canvasHeight = size.height

                            boxes.forEach { box ->

                                if (box.left < 0) return@forEach

                                val rectTopLeft = Offset(
                                    box.left * canvasWidth,
                                    box.top * canvasHeight
                                )
                                val rectSize = Size(
                                    (box.right - box.left) * canvasWidth,
                                    (box.bottom - box.top) * canvasHeight
                                )
                                drawRect(
                                    color = Color.Red,
                                    topLeft = rectTopLeft,
                                    size = rectSize,
                                    style = Stroke(width = 4f)
                                )

                                drawContext.canvas.nativeCanvas.drawText(
                                    box.trackerId.toString(),
                                    rectTopLeft.x + rectSize.width + 20f,
                                    rectTopLeft.y + rectSize.height / 2,
                                    Paint().apply {
                                        color = android.graphics.Color.RED
                                        textSize = 50f
                                        textAlign = Paint.Align.LEFT
                                    }
                                )
                            }


                            if (useBoundingBox && uiPoints.size == 2) {
                                val topLeft = Offset(
                                    minOf(uiPoints[0].x, uiPoints[1].x),
                                    minOf(uiPoints[0].y, uiPoints[1].y)
                                )
                                val bottomRight = Offset(
                                    maxOf(uiPoints[0].x, uiPoints[1].x),
                                    maxOf(uiPoints[0].y, uiPoints[1].y)
                                )

                                drawRect(
                                    color = Color.Green.copy(alpha = 0.5f),
                                    topLeft = topLeft,
                                    size = Size(
                                        width = bottomRight.x - topLeft.x,
                                        height = bottomRight.y - topLeft.y
                                    ),
                                    style = Stroke(
                                        width = 2.dp.toPx(),
                                        cap = StrokeCap.Round,
                                        join = StrokeJoin.Round
                                    )
                                )
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            fun pickColor(trackerIndex: Int) {
                activeTrackerIndex = trackerIndex

                val executor = ContextCompat.getMainExecutor(context)

                val imageCapture = setupViewModel.imageCapture ?: return

                imageCapture.takePicture(
                    executor,
                    object : ImageCapture.OnImageCapturedCallback() {
                        override fun onCaptureSuccess(image: ImageProxy) {
                            val bitmap = image.toBitmap2() // see helper below
                            setupViewModel.frozenBitmap = bitmap
                            setupViewModel.isPickingColor = true
                            image.close()
                        }

                        override fun onError(exception: ImageCaptureException) {
                            // handle error
                        }
                    }
                )
            }

            var showTracker2 by remember { mutableStateOf(false) }
            var showTracker3 by remember { mutableStateOf(false) }

            Column(modifier = Modifier.padding(16.dp)) {

                // ---- Row 1 (Always visible) ----
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Button(onClick = {
                        // TODO: Open color picker
//                        tracker1Color = Color.Red // placeholder
                        pickColor(1)
                    }) {
                        Text("Pick Color")
                    }

                    tracker1Color?.let { color ->
                        Spacer(modifier = Modifier.width(12.dp))
                        ColorPreview(color)
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                // ---- Add Tracker 2 Button (only if not visible) ----
                if (!showTracker2) {
                    Button(onClick = { showTracker2 = true }) {
                        Text("Add 2. tracker")
                    }
                }

                // ---- Row 2 ----
                if (showTracker2) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Button(onClick = {
                            // TODO: Open color picker
//                            tracker2Color = Color.Green // placeholder
                            pickColor(2)
                        }) {
                            Text("Pick Color")
                        }

                        tracker2Color?.let { color ->
                            Spacer(modifier = Modifier.width(12.dp))
                            ColorPreview(color)
                        }

                        Spacer(modifier = Modifier.width(12.dp))

                        Button(onClick = {
                            showTracker2 = false
                            showTracker3 = false
                            tracker2Color = null
                            tracker3Color = null
                        }) {
                            Text("Remove")
                        }
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    // ---- Add Tracker 3 Button (only if tracker 2 exists and 3 doesn't) ----
                    if (!showTracker3) {
                        Button(onClick = { showTracker3 = true }) {
                            Text("Add 3. tracker")
                        }
                    }
                }

                // ---- Row 3 ----
                if (showTracker3) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Button(onClick = {
                            // TODO: Open color picker
//                            tracker3Color = Color.Blue // placeholder
                            pickColor(3)
                        }) {
                            Text("Pick Color")
                        }

                        tracker3Color?.let { color ->
                            Spacer(modifier = Modifier.width(12.dp))
                            ColorPreview(color)
                        }

                        Spacer(modifier = Modifier.width(12.dp))

                        Button(onClick = {
                            showTracker3 = false
                            tracker3Color = null
                        }) {
                            Text("Remove")
                        }
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                Column(
                    modifier = Modifier.padding(horizontal = 16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Row {

                        // Back button always visible
                        OutlinedButton(
                            onClick = back,
                            modifier = Modifier.weight(1f)
                        ) {
                            Text("Back")
                        }

                        Spacer(modifier = Modifier.width(12.dp))

                        Button(onClick = {
                            showStartDialog = true
//                            setupViewModel.storeExperimentSetupAndStart()
                        }) {
                            Text("Start Experiment")
                        }
                    }
                }
            }
        }
    }

    if (showStartDialog) {
        Dialog(
            onDismissRequest = { showStartDialog = false }
        ) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp)
            ) {
                val samplingRateText = setupViewModel.samplingRateText
                OutlinedTextField(
                    value = samplingRateText,
                    onValueChange = { setupViewModel.samplingRateText = it },
                    label = { Text("Sampling rate") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(6.dp),
                    trailingIcon = {
                        Text("/s")
                    }
                )

                val description = setupViewModel.description
                OutlinedTextField(
                    value = description,
                    onValueChange = { setupViewModel.description = it },
                    label = { Text("Experiment description") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(6.dp)
                )

                Row(
                    modifier = Modifier
                        .fillMaxWidth(),
//                        .background(MaterialTheme.colorScheme.surfaceContainer),
                    horizontalArrangement = Arrangement.End,
                    verticalAlignment = Alignment.Bottom
                ) {
                    TextButton(
                        onClick = { showStartDialog = false },
                        modifier = Modifier.padding(8.dp)
                    ) {
                        Text(text = "Cancel")
                    }
                    TextButton(
                        onClick = {
                            setupViewModel.storeExperimentSetupAndStart()
                        },
                        modifier = Modifier.padding(8.dp),
                        enabled = samplingRateText.toIntOrNull()?.let { it in 1..60 } ?: false,
//                        colors = ButtonDefaults.textButtonColors(
//                            contentColor = MaterialTheme.colorScheme.error
//                        )
                    ) {
                        Text(text = "Start")
                    }
                }
            }
        }
    }
}

@Composable
fun ColorPreview(color: Color) {
    Box(
        modifier = Modifier
            .size(24.dp)
            .background(color = color, shape = CircleShape)
            .border(1.dp, Color.Black, CircleShape)
    )
}