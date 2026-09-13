package de.tudarmstadt.physics.trackingplot.ui

import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.dropUnlessStarted
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavBackStackEntry
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.navigation
import androidx.navigation.compose.rememberNavController
import androidx.navigation.toRoute
import de.tudarmstadt.physics.trackingplot.TrackingPlotApp
import de.tudarmstadt.physics.trackingplot.db.ExperimentDatabase
import de.tudarmstadt.physics.trackingplot.ui.common.CameraPermissionGate
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

@Serializable object ExperimentsOverviewRoute
@Serializable object SetupRoute
@Serializable object RulerSetupRoute
@Serializable object BoundingBoxSetupRoute
@Serializable object TrackerSetupRoute
@Serializable data class LiveExperimentRoute(val experimentId: Long)
@Serializable data class PastExperimentRoute(val experimentId: Long)

@Composable
fun Navigation() {
    val navController = rememberNavController()
    val context = LocalContext.current
    val db = remember { ExperimentDatabase.getInstance(context) }
    val app = remember { TrackingPlotApp.from(context) }

    NavHost(
        navController = navController,
        startDestination = ExperimentsOverviewRoute,
        enterTransition = { EnterTransition.None },
        exitTransition = { ExitTransition.None }
    ) {
        composable<ExperimentsOverviewRoute> {
            ExperimentsOverviewScreen(
                toSetup = dropUnlessStarted { navController.navigate(SetupRoute) },
                toPastExperiment = { experimentId ->
                    navController.navigateIfStarted(it, PastExperimentRoute(experimentId))
                },
                viewModel = viewModel(factory = viewModelFactory { ExperimentsOverviewViewModel(db) })
            )
        }

        navigation<SetupRoute>(startDestination = RulerSetupRoute) {
            composable<RulerSetupRoute> { entry ->
                CameraPermissionGate {
                    RulerSetupScreen(
                        toNextStep = dropUnlessStarted { navController.navigate(BoundingBoxSetupRoute) },
                        setupViewModel = navController.setupViewModel(entry, db)
                    )
                }
            }
            composable<BoundingBoxSetupRoute> { entry ->
                CameraPermissionGate {
                    BoundingBoxSetupScreen(
                        back = dropUnlessStarted { navController.navigateUp() },
                        toNextStep = dropUnlessStarted { navController.navigate(TrackerSetupRoute) },
                        setupViewModel = navController.setupViewModel(entry, db)
                    )
                }
            }
            composable<TrackerSetupRoute> { entry ->
                CameraPermissionGate {
                    TrackerSetupScreen(
                        back = dropUnlessStarted { navController.navigateUp() },
                        toLiveExperiment = { experimentId ->
                            if (entry.isStarted()) {
                                navController.navigate(LiveExperimentRoute(experimentId)) {
                                    popUpTo<SetupRoute> { inclusive = true }
                                }
                            }
                        },
                        setupViewModel = navController.setupViewModel(entry, db)
                    )
                }
            }
        }

        composable<LiveExperimentRoute> { entry ->
            val experimentId = entry.toRoute<LiveExperimentRoute>().experimentId
            CameraPermissionGate {
                LiveExperimentScreen(
                    onLeave = dropUnlessStarted { navController.popBackStack() },
                    onShowResults = {
                        if (entry.isStarted()) {
                            navController.navigate(PastExperimentRoute(experimentId)) {
                                popUpTo<LiveExperimentRoute> { inclusive = true }
                            }
                        }
                    },
                    viewModel = viewModel(
                        factory = viewModelFactory {
                            LiveExperimentViewModel(experimentId, db, app.appScope)
                        }
                    )
                )
            }
        }

        composable<PastExperimentRoute> { entry ->
            val experimentId = entry.toRoute<PastExperimentRoute>().experimentId
            PastExperimentScreen(
                viewModel = viewModel(factory = viewModelFactory { PastExperimentViewModel(experimentId, db) }),
                onBack = dropUnlessStarted { navController.popBackStack() }
            )
        }
    }
}

/** The [SetupViewModel] shared by all screens of the setup graph. */
@Composable
private fun NavHostController.setupViewModel(entry: NavBackStackEntry, db: ExperimentDatabase): SetupViewModel {
    val graphEntry = remember(entry) { getBackStackEntry<SetupRoute>() }
    return viewModel(
        viewModelStoreOwner = graphEntry,
        factory = viewModelFactory { SetupViewModel(db) }
    )
}

private fun NavBackStackEntry.isStarted(): Boolean =
    lifecycle.currentState.isAtLeast(androidx.lifecycle.Lifecycle.State.STARTED)

private fun NavHostController.navigateIfStarted(entry: NavBackStackEntry, route: Any) {
    if (entry.isStarted()) navigate(route)
}
