package de.tudarmstadt.physics.trackingplot.ui.experiments_overview

import androidx.compose.runtime.mutableStateListOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import de.tudarmstadt.physics.trackingplot.Experiment
import de.tudarmstadt.physics.trackingplot.db.ExperimentDatabase
import kotlinx.coroutines.launch

class ExperimentsOverviewViewModel(
    private val db: ExperimentDatabase
): ViewModel() {

    private val _experiments = mutableStateListOf<Experiment>()
    val experiments = _experiments as List<Experiment>

    init {
        loadExperiments()
    }

    fun loadExperiments() {
        viewModelScope.launch {
            val experiments = db.withTransaction(readOnly = true) {
                getExperiments()
            }
            _experiments.clear()
            _experiments.addAll(experiments)
        }
    }
}