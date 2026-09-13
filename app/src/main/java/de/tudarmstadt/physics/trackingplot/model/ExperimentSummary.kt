package de.tudarmstadt.physics.trackingplot.model

data class ExperimentSummary(
    val experimentId: Long,
    val description: String,
    /** Creation time as epoch milliseconds; 0 for experiments recorded before this was stored. */
    val createdAt: Long,
    val trackerCount: Int,
    val sampleCount: Long
)
