package de.tudarmstadt.physics.trackingplot.ui

import android.util.Rational
import androidx.camera.compose.CameraXViewfinder
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceRequest
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.lifecycle.awaitInstance
import androidx.camera.viewfinder.compose.MutableCoordinateTransformer
import androidx.camera.viewfinder.core.ImplementationMode
import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.transformable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.dropUnlessStarted
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.navigation
import androidx.navigation.compose.rememberNavController
import androidx.navigation.toRoute
import de.tudarmstadt.physics.trackingplot.tracker2.NativeTracker
import de.tudarmstadt.physics.trackingplot.tracking.CameraAnalyzer
import de.tudarmstadt.physics.trackingplot.ui.experiments_overview.ExperimentsOverviewScreen
import de.tudarmstadt.physics.trackingplot.ui.experiments_overview.ExperimentsOverviewViewModel
import de.tudarmstadt.physics.trackingplot.ui.live_experiment.LiveExperimentScreen
import de.tudarmstadt.physics.trackingplot.ui.live_experiment.LiveExperimentViewModel
import de.tudarmstadt.physics.trackingplot.ui.past_experiment.PastExperimentScreen
import de.tudarmstadt.physics.trackingplot.ui.past_experiment.PastExperimentViewModel
import de.tudarmstadt.physics.trackingplot.ui.setup.SetupViewModel
import de.tudarmstadt.physics.trackingplot.ui.setup.bounding_box.BoundingBoxSetupScreen
import de.tudarmstadt.physics.trackingplot.ui.setup.ruler.RulerSetupScreen
import de.tudarmstadt.physics.trackingplot.ui.setup.tracker.TrackerSetupScreen
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.serialization.Serializable
import java.util.concurrent.Executors

//sealed class Screen(val route: String) {
//    data object RulerCalibration : Screen("ruler")
//    data object TrackingArea : Screen("area")
//    data object TrackingConfiguration : Screen("configuration")
//    data object LiveTracking : Screen("tracking")
//}
//
//@Composable
//fun AppNavigation(modifier: Modifier = Modifier) {
//    val navController = rememberNavController()
//
//    NavHost(navController, startDestination = Screen.RulerCalibration.route) {
//        composable(Screen.RulerCalibration.route) {
//            val context = LocalContext.current
//            val lifecycleOwner = LocalLifecycleOwner.current
//            val selector = CameraSelector.DEFAULT_BACK_CAMERA
//
//            val surfaceRequests = remember { MutableStateFlow<SurfaceRequest?>(null) }
//            val surfaceRequest by surfaceRequests.collectAsState(initial = null)
//
//            LaunchedEffect(Unit) {
//                val provider = ProcessCameraProvider.awaitInstance(context)
//                val preview = Preview.Builder().build().apply {
//                    setSurfaceProvider { req -> surfaceRequests.value = req }
//                }
//                provider.unbindAll()
//                provider.bindToLifecycle(lifecycleOwner, selector, preview)
//            }
//
//            surfaceRequest?.let {
//                CameraWithMarkersRuler(
//                    surfaceRequest = it,
//                    onDone = { points ->
//                        navController.navigate(Screen.TrackingArea.route)
//                    }
//                )
//            }
////            RulerCalibrationScreen(
////                onDone = { p1, p2 ->
////                    // save to viewmodel / shared state
////                    navController.navigate(Screen.TrackingArea.route)
////                }
////            )
//        }
//        composable(Screen.TrackingArea.route) {
//            val context = LocalContext.current
//            val lifecycleOwner = LocalLifecycleOwner.current
//            val selector = CameraSelector.DEFAULT_BACK_CAMERA
//
//            val surfaceRequests = remember { MutableStateFlow<SurfaceRequest?>(null) }
//            val surfaceRequest by surfaceRequests.collectAsState(initial = null)
//
//            var rect by remember { mutableStateOf<Rect?>(null) }
//
//            LaunchedEffect(Unit) {
//                val provider = ProcessCameraProvider.awaitInstance(context)
//                val preview = Preview.Builder().build().apply {
//                    setSurfaceProvider { req -> surfaceRequests.value = req }
//                }
//                val analyzer = ImageAnalysis.Builder()
//                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//                    .setOutputImageFormat(ImageAnalysis.OUTF)
////                    .build()
////                    .apply {
////                        setAnalyzer(
////                            Executors.newSingleThreadExecutor(),
////                            CameraAnalyzer { rect = it }
////                        )
////                    }
//                provider.unbindAll()
//                provider.bindToLifecycle(lifecycleOwner, selector, preview)
//            }
//
//            surfaceRequest?.let {
//                CameraWithMarkersBox(
//                    surfaceRequest = it,
//                    onDone = { points ->
//                        navController.navigate(Screen.TrackingConfiguration.route)
//                    },
//                    rect = rect
//                )
//            }
////            TrackingAreaSelectionScreen(
////                onDone = { rect ->
////                    // save rect
////                    navController.navigate(Screen.LiveTracking.route)
////                }
////            )
//        }
//        composable(Screen.TrackingConfiguration.route) {
//            val context = LocalContext.current
//            val lifecycleOwner = LocalLifecycleOwner.current
//            val selector = CameraSelector.DEFAULT_BACK_CAMERA
//
//            val surfaceRequests = remember { MutableStateFlow<SurfaceRequest?>(null) }
//            val surfaceRequest by surfaceRequests.collectAsState(initial = null)
//
//            LaunchedEffect(Unit) {
//                val provider = ProcessCameraProvider.awaitInstance(context)
//                val preview = Preview.Builder().build().apply {
//                    setSurfaceProvider { req -> surfaceRequests.value = req }
//                }
//                provider.unbindAll()
//                provider.bindToLifecycle(lifecycleOwner, selector, preview)
//            }
//
//            surfaceRequest?.let {
//                CameraWithConfiguration(
//                    surfaceRequest = it,
//                    onDone = { points ->
//                        navController.navigate(Screen.LiveTracking.route)
//                    }
//                )
//            }
////            TrackingAreaSelectionScreen(
////                onDone = { rect ->
////                    // save rect
////                    navController.navigate(Screen.LiveTracking.route)
////                }
////            )
//        }
//        composable(Screen.LiveTracking.route) {
//            MainScreen()
////            LiveTrackingScreen()
//        }
//    }
//}



@Composable
fun AppNavigation() {
    val navController = rememberNavController()

    NavHost(
        navController = navController,
        startDestination = ExperimentsOverviewRoute,
        enterTransition = { EnterTransition.None },
        exitTransition = { ExitTransition.None }
    ) {
        composable<ExperimentsOverviewRoute> {
            val lifecycleOwner = LocalLifecycleOwner.current
            ExperimentsOverviewScreen(
                toSetup = dropUnlessStarted { navController.navigate(SetupRoute) },
                toPastExperiment = { experimentId ->
                    //explicit (and equal) version of dropUnlessStarted
                    if (lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) {
                        navController.navigate(PastExperimentRoute(experimentId))
                    }
                },
                viewModel = viewModel<ExperimentsOverviewViewModel>()
            )
        }

        navigation<SetupRoute>(startDestination = RulerSetupRoute) {
            composable<RulerSetupRoute> { entry ->
                val parentEntry = remember(entry) { navController.getBackStackEntry(RulerSetupRoute) }
//                val setupViewModel = viewModel<SetupViewModel>(viewModelStoreOwner = parentEntry)
                val setupViewModel = viewModel<SetupViewModel>(
                    viewModelStoreOwner = parentEntry,
                    factory = viewModelFactory {
                        SetupViewModel(
                            nativeTracker = NativeTracker()
                        )
                    }
                )
                RulerSetupScreen(
                    toNextStep = dropUnlessStarted {
                        navController.navigate(BoundingBoxSetupRoute)
                    },
                    setupViewModel = setupViewModel
                )
                Button(
                    onClick = dropUnlessStarted {
                        navController.navigate(LiveExperimentRoute(1234))
                    },
                    modifier = Modifier.padding(32.dp)
                ) {
                    Text("TEMP NAV BUTTON")
                }
            }
            composable<BoundingBoxSetupRoute> { entry ->
                val parentEntry = remember(entry) { navController.getBackStackEntry(RulerSetupRoute) }
//                val setupViewModel = viewModel<SetupViewModel>(viewModelStoreOwner = parentEntry)
                val setupViewModel = viewModel<SetupViewModel>(
                    viewModelStoreOwner = parentEntry,
                    factory = viewModelFactory {
                        SetupViewModel(
                            nativeTracker = NativeTracker()
                        )
                    }
                )
                BoundingBoxSetupScreen(
                    back = dropUnlessStarted { navController.navigateUp() },
                    toNextStep = dropUnlessStarted {
                        navController.navigate(TrackerSetupRoute)
                    },
                    skip = {},
                    setupViewModel = setupViewModel
                )
            }
            composable<TrackerSetupRoute> { entry ->
                val parentEntry = remember(entry) { navController.getBackStackEntry(RulerSetupRoute) }
                val setupViewModel = viewModel<SetupViewModel>(
                    viewModelStoreOwner = parentEntry,
                    factory = viewModelFactory {
                        SetupViewModel(
                            nativeTracker = NativeTracker()
                        )
                    }
                )
                val lifecycleOwner = LocalLifecycleOwner.current
                TrackerSetupScreen(
                    toLiveExperiment = { experimentId ->
                        //explicit (and equal) version of dropUnlessStarted
                        if (lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) {
                            navController.navigate(LiveExperimentRoute(experimentId))
                        }
                    },
                    setupViewModel = setupViewModel
                )
            }
        }

        composable<LiveExperimentRoute> { entry ->
            val experimentId = entry.toRoute<LiveExperimentRoute>().experimentId
            LiveExperimentScreen(
                onAbortExperiment = dropUnlessStarted { navController.navigateUp() },
                viewModel = viewModel(
                    factory = viewModelFactory {
                        LiveExperimentViewModel(
                            experimentId = experimentId
                        )
                    }
                )
            )
            val context = LocalContext.current
            val lifecycleOwner = LocalLifecycleOwner.current
            val selector = CameraSelector.DEFAULT_BACK_CAMERA

            val surfaceRequests = remember { MutableStateFlow<SurfaceRequest?>(null) }
            val surfaceRequest by surfaceRequests.collectAsState(initial = null)

            var tmpNormalized by remember { mutableStateOf<Offset?>(null) }

            LaunchedEffect(Unit) {
                val provider = ProcessCameraProvider.awaitInstance(context)
                val preview = Preview.Builder().build().apply {
                    setSurfaceProvider { req -> surfaceRequests.value = req }
                }
//                val analyzer = ImageAnalysis.Builder()
//                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//                    .setOutputImageFormat(ImageAnalysis.OUTF)
////                    .build()
////                    .apply {
////                        setAnalyzer(
////                            Executors.newSingleThreadExecutor(),
////                            CameraAnalyzer { rect = it }
////                        )
////                    }
                val analyzer = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .apply {
                        setAnalyzer(
                            Executors.newSingleThreadExecutor(),
                            CameraAnalyzer(onResult = {})
                        )
                        if (1==1) return@apply
                        setAnalyzer(
                            Executors.newSingleThreadExecutor(),
                            object : ImageAnalysis.Analyzer {
                                override fun analyze(image: ImageProxy) {
                                    val normalized = tmpNormalized
                                    if (normalized == null) {
                                        image.close()
                                        return
                                    }

                                    tmpNormalized = null

                                    val crop = image.cropRect

                                    println("crop")
                                    println(crop)

                                    val x = crop.left + (normalized.x * crop.width()).toInt()
                                    val y = crop.top + (normalized.y * crop.height()).toInt()

                                    val (finalX, finalY) = when (image.imageInfo.rotationDegrees) {
                                        90 -> image.height - y to x
                                        180 -> image.width - x to image.height - y
                                        270 -> y to image.width - x
                                        else -> x to y
                                    }
                                    val (finalWidth, finalHeight) = when (image.imageInfo.rotationDegrees) {
                                        90, 270 -> image.height to image.width
                                        else -> image.width to image.height
                                    }

                                    println("COORDS: $finalX $finalY out of $finalWidth $finalHeight")

                                    image.close()
                                }
                            }
                        )
                    }

                provider.unbindAll()
                provider.bindToLifecycle(lifecycleOwner, selector, preview, analyzer)
//                provider.unbindAll()
//                provider.bindToLifecycle(lifecycleOwner, selector, )
            }

            surfaceRequest?.let {
                BoxWithConstraints(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color.Black),
                    contentAlignment = Alignment.TopCenter
                ) {
//                    val aspectRatio = 3 to 4
                    val aspectRatio = 9 to 16
                    val maxAspectRatio: Float = maxWidth / maxHeight
                    val aspectRatioFloat: Float = aspectRatio.first.toFloat() / aspectRatio.second
                    val shouldUseMaxWidth = maxAspectRatio <= aspectRatioFloat
                    val width = if (shouldUseMaxWidth) maxWidth else maxHeight * aspectRatioFloat
                    val height = if (!shouldUseMaxWidth) maxHeight else maxWidth / aspectRatioFloat

                    Box(
                        modifier = Modifier
//                            .width(width)
//                            .height(height)
//                            .transformable()
                            .clip(RoundedCornerShape(16.dp))
                    ) {
                        val implementationMode = ImplementationMode.EXTERNAL

                        val coordinateTransformer = remember { MutableCoordinateTransformer() }
                        CameraXViewfinder(
                            modifier = Modifier
                                .fillMaxSize()
                                .pointerInput(Unit) {
                                    detectTapGestures(
                                        onDoubleTap = { offset ->
                                        },
                                        onTap = { offset ->
                                            with(coordinateTransformer) {
                                                val surfaceCoords = offset.transform()
                                                val surfaceResolution = it.resolution

                                                println("surface coords")
                                                println("${surfaceCoords.x} ${surfaceCoords.y}")

                                                val normalized = Offset(
                                                    surfaceCoords.x / surfaceResolution.width,
                                                    surfaceCoords.y / surfaceResolution.height
                                                )
                                                println("normalized coords")
                                                println("${normalized.x} ${normalized.y}")

                                                tmpNormalized = normalized
                                            }
                                        }
                                    )
                                },
                            surfaceRequest = it,
                            implementationMode = implementationMode,
                            coordinateTransformer = coordinateTransformer,
                            contentScale = ContentScale.Fit
                        )
                    }
                }
            }
        }

        composable<PastExperimentRoute> { entry ->
            val experimentId = entry.toRoute<PastExperimentRoute>().experimentId
            PastExperimentScreen(
                viewModel = viewModel(
                    factory = viewModelFactory {
                        PastExperimentViewModel(
                            experimentId = experimentId
                        )
                    }
                )
            )
        }
    }
}

@Serializable object ExperimentsOverviewRoute
@Serializable object SetupRoute
@Serializable object RulerSetupRoute
@Serializable object BoundingBoxSetupRoute
@Serializable object TrackerSetupRoute
//@Serializable object LiveExperimentRoute
@Serializable class LiveExperimentRoute(val experimentId: Long)
@Serializable class PastExperimentRoute(val experimentId: Long)
