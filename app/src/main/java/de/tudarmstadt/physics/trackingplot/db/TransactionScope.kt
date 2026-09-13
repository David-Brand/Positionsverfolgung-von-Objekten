package de.tudarmstadt.physics.trackingplot.db

import android.content.ContentValues
import android.database.sqlite.SQLiteDatabase
import de.tudarmstadt.physics.trackingplot.model.ColorTrackerConfig
import de.tudarmstadt.physics.trackingplot.model.DistanceUnit
import de.tudarmstadt.physics.trackingplot.model.ExperimentSummary
import de.tudarmstadt.physics.trackingplot.model.MeasurementSample
import de.tudarmstadt.physics.trackingplot.model.Point2D
import de.tudarmstadt.physics.trackingplot.model.Roi
import de.tudarmstadt.physics.trackingplot.model.Ruler
import de.tudarmstadt.physics.trackingplot.model.TrackingConfig

/** Query API available inside [ExperimentDatabase.read] and [ExperimentDatabase.write]. */
class TransactionScope internal constructor(private val db: SQLiteDatabase) {

    fun getExperiments(): List<ExperimentSummary> {
        val query = "SELECT e.ExperimentId, e.Description, e.CreatedAt" +
            ", (SELECT COUNT(*) FROM Tracker t WHERE t.ExperimentId = e.ExperimentId)" +
            ", (SELECT COUNT(*) FROM Measurement m WHERE m.ExperimentId = e.ExperimentId)" +
            " FROM Experiment AS e" +
            " ORDER BY e.ExperimentId DESC"
        db.rawQuery(query, null).use { cursor ->
            val results = ArrayList<ExperimentSummary>(cursor.count)
            while (cursor.moveToNext()) {
                results.add(
                    ExperimentSummary(
                        experimentId = cursor.getLong(0),
                        description = cursor.getString(1),
                        createdAt = cursor.getLong(2),
                        trackerCount = cursor.getInt(3),
                        sampleCount = cursor.getLong(4)
                    )
                )
            }
            return results
        }
    }

    fun getExperiment(experimentId: Long): ExperimentSummary? {
        val query = "SELECT e.ExperimentId, e.Description, e.CreatedAt" +
            ", (SELECT COUNT(*) FROM Tracker t WHERE t.ExperimentId = e.ExperimentId)" +
            ", (SELECT COUNT(*) FROM Measurement m WHERE m.ExperimentId = e.ExperimentId)" +
            " FROM Experiment AS e WHERE e.ExperimentId = ?"
        db.rawQuery(query, arrayOf(experimentId.toString())).use { cursor ->
            if (!cursor.moveToFirst()) return null
            return ExperimentSummary(
                experimentId = cursor.getLong(0),
                description = cursor.getString(1),
                createdAt = cursor.getLong(2),
                trackerCount = cursor.getInt(3),
                sampleCount = cursor.getLong(4)
            )
        }
    }

    /** Returns null when the experiment does not exist. */
    fun getTrackingConfig(experimentId: Long): TrackingConfig? {
        val query = "SELECT" +
            " e.SamplingRate, e.ZoomRatio" +
            ", r.P1X, r.P1Y, r.P2X, r.P2Y, r.Distance, r.Unit" +
            ", roi.P1X, roi.P1Y, roi.P2X, roi.P2Y" +
            " FROM Experiment AS e" +
            " LEFT JOIN Ruler AS r ON e.ExperimentId = r.ExperimentId" +
            " LEFT JOIN Roi AS roi ON e.ExperimentId = roi.ExperimentId" +
            " WHERE e.ExperimentId = ?"

        val (samplingRate, zoomRatio, ruler, roi) = db.rawQuery(query, arrayOf(experimentId.toString())).use { cursor ->
            if (!cursor.moveToFirst()) return null

            val ruler = if (!cursor.isNull(2)) {
                val unit = DistanceUnit.fromSymbolOrNull(cursor.getString(7))
                if (unit == null) {
                    null
                } else {
                    Ruler(
                        p1 = Point2D(cursor.getFloat(2), cursor.getFloat(3)),
                        p2 = Point2D(cursor.getFloat(4), cursor.getFloat(5)),
                        realDistance = cursor.getFloat(6),
                        unit = unit
                    )
                }
            } else null

            val roi = if (!cursor.isNull(8)) {
                Roi(
                    p1 = Point2D(cursor.getFloat(8), cursor.getFloat(9)),
                    p2 = Point2D(cursor.getFloat(10), cursor.getFloat(11))
                )
            } else null

            ConfigRow(cursor.getInt(0), cursor.getFloat(1), ruler, roi)
        }

        val trackers = db.rawQuery(
            "SELECT TrackerIndex, Color, Tolerance FROM Tracker WHERE ExperimentId = ? ORDER BY TrackerIndex",
            arrayOf(experimentId.toString())
        ).use { cursor ->
            val results = ArrayList<ColorTrackerConfig>(cursor.count)
            while (cursor.moveToNext()) {
                results.add(
                    ColorTrackerConfig(
                        color = cursor.getInt(1),
                        tolerance = cursor.getFloat(2).toInt()
                    )
                )
            }
            results
        }

        return TrackingConfig(
            roi = roi,
            ruler = ruler,
            trackers = trackers,
            samplingRate = samplingRate,
            zoomRatio = if (zoomRatio.isFinite() && zoomRatio > 0f) zoomRatio else 1f
        )
    }

    private data class ConfigRow(val samplingRate: Int, val zoomRatio: Float, val ruler: Ruler?, val roi: Roi?)

    /** All measurements of an experiment grouped by tracker index, ordered by time. */
    fun getMeasurements(experimentId: Long): Map<Int, List<MeasurementSample>> {
        val query = "SELECT TrackerIndex, Timestamp, CentroidX, CentroidY" +
            " FROM Measurement WHERE ExperimentId = ?" +
            " ORDER BY TrackerIndex, Timestamp"
        db.rawQuery(query, arrayOf(experimentId.toString())).use { cursor ->
            val results = LinkedHashMap<Int, MutableList<MeasurementSample>>()
            while (cursor.moveToNext()) {
                val trackerIndex = cursor.getInt(0)
                results.getOrPut(trackerIndex) { ArrayList() }.add(
                    MeasurementSample(
                        trackerIndex = trackerIndex,
                        timestampMs = cursor.getLong(1),
                        centroidX = cursor.getDouble(2),
                        centroidY = cursor.getDouble(3)
                    )
                )
            }
            return results
        }
    }

    /** Largest stored timestamp of the experiment, or null when it has no measurements yet. */
    fun getLastTimestamp(experimentId: Long): Long? {
        db.rawQuery(
            "SELECT MAX(Timestamp) FROM Measurement WHERE ExperimentId = ?",
            arrayOf(experimentId.toString())
        ).use { cursor ->
            if (!cursor.moveToFirst() || cursor.isNull(0)) return null
            return cursor.getLong(0)
        }
    }

    fun addExperiment(trackingConfig: TrackingConfig, description: String, createdAt: Long): Long {
        val values = ContentValues().apply {
            put("Description", description)
            put("CreatedAt", createdAt)
            put("SamplingRate", trackingConfig.samplingRate)
            put("ZoomRatio", trackingConfig.zoomRatio)
        }
        val experimentId = db.insertOrThrow("Experiment", null, values)

        trackingConfig.ruler?.let { ruler ->
            values.clear()
            values.put("ExperimentId", experimentId)
            values.put("P1X", ruler.p1.x)
            values.put("P1Y", ruler.p1.y)
            values.put("P2X", ruler.p2.x)
            values.put("P2Y", ruler.p2.y)
            values.put("Distance", ruler.realDistance)
            values.put("Unit", ruler.unit.symbol)
            db.insertOrThrow("Ruler", null, values)
        }

        trackingConfig.roi?.let { roi ->
            values.clear()
            values.put("ExperimentId", experimentId)
            values.put("P1X", roi.p1.x)
            values.put("P1Y", roi.p1.y)
            values.put("P2X", roi.p2.x)
            values.put("P2Y", roi.p2.y)
            db.insertOrThrow("Roi", null, values)
        }

        trackingConfig.trackers.forEachIndexed { index, tracker ->
            values.clear()
            values.put("ExperimentId", experimentId)
            values.put("TrackerIndex", index)
            values.put("Color", tracker.color)
            values.put("Tolerance", tracker.tolerance.toFloat())
            db.insertOrThrow("Tracker", null, values)
        }

        return experimentId
    }

    fun deleteExperiment(experimentId: Long): Boolean {
        return db.delete("Experiment", "ExperimentId = ?", arrayOf(experimentId.toString())) > 0
    }

    /**
     * Inserts measurements in one batch. A sample whose (tracker, timestamp) key already exists
     * is ignored rather than failing the whole batch.
     */
    fun addMeasurements(experimentId: Long, samples: List<MeasurementSample>) {
        if (samples.isEmpty()) return
        db.compileStatement(
            "INSERT OR IGNORE INTO Measurement (ExperimentId, TrackerIndex, Timestamp, CentroidX, CentroidY)" +
                " VALUES (?, ?, ?, ?, ?)"
        ).use { statement ->
            for (sample in samples) {
                statement.clearBindings()
                statement.bindLong(1, experimentId)
                statement.bindLong(2, sample.trackerIndex.toLong())
                statement.bindLong(3, sample.timestampMs)
                statement.bindDouble(4, sample.centroidX)
                statement.bindDouble(5, sample.centroidY)
                statement.executeInsert()
            }
        }
    }
}
