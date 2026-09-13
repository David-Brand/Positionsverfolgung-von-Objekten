package de.tudarmstadt.physics.trackingplot.db

import android.util.Log
import de.tudarmstadt.physics.trackingplot.model.MeasurementSample
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ChannelResult
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicLong

/**
 * Persists measurement samples in batches on a background thread.
 *
 * Samples are queued with [submit] (cheap, never blocks) and written in transactions of up to
 * [MAX_BATCH] rows every [FLUSH_DELAY_MS]. The writer runs in [scope], which should outlive the
 * screen that produces the samples, so that nothing is lost when the user leaves; [close] stops
 * accepting samples and the remaining queue is still written.
 */
class MeasurementWriter(
    private val db: ExperimentDatabase,
    private val experimentId: Long,
    scope: CoroutineScope
) {
    private val channel = Channel<MeasurementSample>(capacity = QUEUE_CAPACITY)
    private val _error = MutableStateFlow<Throwable?>(null)

    /** Last persistence error, if any. */
    val error: StateFlow<Throwable?> = _error

    private val dropped = AtomicLong(0)

    /** Number of samples that could not be queued because the database fell too far behind. */
    val droppedCount: Long get() = dropped.get()

    private val job: Job = scope.launch(Dispatchers.IO) { drain() }

    fun submit(samples: List<MeasurementSample>) {
        for (sample in samples) {
            val result: ChannelResult<Unit> = channel.trySend(sample)
            if (result.isFailure) {
                if (result.isClosed) return
                dropped.incrementAndGet()
            }
        }
    }

    /** Stops accepting samples; queued samples are still written. */
    fun close() {
        channel.close()
    }

    /** [close] and wait until everything queued has been written. */
    suspend fun closeAndAwait() {
        close()
        job.join()
    }

    private suspend fun drain() {
        val batch = ArrayList<MeasurementSample>(MAX_BATCH)
        for (first in channel) {
            batch.clear()
            batch.add(first)
            delay(FLUSH_DELAY_MS)
            while (batch.size < MAX_BATCH) {
                batch.add(channel.tryReceive().getOrNull() ?: break)
            }
            try {
                db.write { addMeasurements(experimentId, batch) }
            } catch (e: Exception) {
                // Keep draining so that a transient error does not stop persistence for good.
                Log.e(TAG, "Could not store ${batch.size} measurements", e)
                _error.value = e
            }
        }
    }

    private companion object {
        const val TAG = "MeasurementWriter"
        const val QUEUE_CAPACITY = 20_000
        const val MAX_BATCH = 512
        const val FLUSH_DELAY_MS = 250L
    }
}
