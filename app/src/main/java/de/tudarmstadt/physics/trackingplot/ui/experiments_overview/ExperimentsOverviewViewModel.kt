package de.tudarmstadt.physics.trackingplot.ui.experiments_overview

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import de.tudarmstadt.physics.trackingplot.db.DatabaseException
import de.tudarmstadt.physics.trackingplot.db.ExperimentDatabase
import de.tudarmstadt.physics.trackingplot.model.ExperimentSummary
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class ExperimentsOverviewViewModel(
    private val db: ExperimentDatabase
) : ViewModel() {

    private val _experiments = MutableStateFlow<List<ExperimentSummary>?>(null)

    /** Null while loading. */
    val experiments: StateFlow<List<ExperimentSummary>?> = _experiments

    private val _error = MutableStateFlow<Throwable?>(null)
    val error: StateFlow<Throwable?> = _error

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            try {
                _experiments.value = db.read { getExperiments() }
                _error.value = null
            } catch (e: DatabaseException) {
                _error.value = e
                if (_experiments.value == null) _experiments.value = emptyList()
            }
        }
    }
}
