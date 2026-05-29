package de.tudarmstadt.physics.trackingplot.ui

import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.dropUnlessStarted
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.navigation
import androidx.navigation.compose.rememberNavController
import androidx.navigation.toRoute
import de.tudarmstadt.physics.trackingplot.TrackingPlotApp
import de.tudarmstadt.physics.trackingplot.db.ExperimentDatabase
import de.tudarmstadt.physics.trackingplot.tracker2.NativeTracker
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
import kotlinx.serialization.Serializable

@Composable
fun Navigation() {
    val navController = rememberNavController()

    val context = LocalContext.current

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
                viewModel = viewModel(
                    factory = viewModelFactory {
                        ExperimentsOverviewViewModel(
                            db = ExperimentDatabase.getInstance(context)
                        )
                    }
                )
            )
        }

        navigation<SetupRoute>(startDestination = RulerSetupRoute) {
            composable<RulerSetupRoute> { entry ->
                val parentEntry = remember(entry) { navController.getBackStackEntry(RulerSetupRoute) }
                val setupViewModel = viewModel<SetupViewModel>(
                    viewModelStoreOwner = parentEntry,
                    factory = viewModelFactory {
                        SetupViewModel(
                            nativeTracker = NativeTracker(),
                            trackingSession = TrackingPlotApp.trackingModule.trackingSession,
                            db = ExperimentDatabase.getInstance(context)
                        )
                    }
                )
                RulerSetupScreen(
                    toNextStep = dropUnlessStarted {
                        navController.navigate(BoundingBoxSetupRoute)
                    },
                    setupViewModel = setupViewModel
                )
            }
            composable<BoundingBoxSetupRoute> { entry ->
                val parentEntry = remember(entry) { navController.getBackStackEntry(RulerSetupRoute) }
                val setupViewModel = viewModel<SetupViewModel>(
                    viewModelStoreOwner = parentEntry,
                    factory = viewModelFactory {
                        SetupViewModel(
                            nativeTracker = NativeTracker(),
                            trackingSession = TrackingPlotApp.trackingModule.trackingSession,
                            db = ExperimentDatabase.getInstance(context)
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
                            nativeTracker = NativeTracker(),
                            trackingSession = TrackingPlotApp.trackingModule.trackingSession,
                            db = ExperimentDatabase.getInstance(context)
                        )
                    }
                )
                val lifecycleOwner = LocalLifecycleOwner.current
                TrackerSetupScreen(
                    back = dropUnlessStarted { navController.navigateUp() },
                    toLiveExperiment = { experimentId ->
                        //explicit (and equal) version of dropUnlessStarted
                        if (lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) {
                            navController.navigate(LiveExperimentRoute(experimentId)) {
                                popUpTo(SetupRoute) {
                                    inclusive = true
                                }
                            }
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
                            experimentId = experimentId,
                            trackingSession = TrackingPlotApp.trackingModule.trackingSession,
                            nativeTracker = NativeTracker(),
                            db = ExperimentDatabase.getInstance(context)
                        )
                    }
                )
            )
        }

        composable<PastExperimentRoute> { entry ->
            val experimentId = entry.toRoute<PastExperimentRoute>().experimentId
            PastExperimentScreen(
                viewModel = viewModel(
                    factory = viewModelFactory {
                        PastExperimentViewModel(
                            experimentId = experimentId,
                            db = ExperimentDatabase.getInstance(context)
                        )
                    }
                ),
                onDelete = dropUnlessStarted { navController.navigateUp() }
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
