package de.tudarmstadt.physics.trackingplot.db

import android.content.ContentValues
import android.database.sqlite.SQLiteDatabase
import de.tudarmstadt.physics.trackingplot.DistanceUnit
import de.tudarmstadt.physics.trackingplot.Experiment
import de.tudarmstadt.physics.trackingplot.tracker2.ColorTrackerConfig
import de.tudarmstadt.physics.trackingplot.tracker2.Point2D
import de.tudarmstadt.physics.trackingplot.tracker2.Roi
import de.tudarmstadt.physics.trackingplot.tracker2.Ruler
import de.tudarmstadt.physics.trackingplot.tracker2.TrackingConfig
import de.tudarmstadt.physics.trackingplot.ui.setup.tracker.NormalizedBox
import de.tudarmstadt.physics.trackingplot.ui.setup.tracker.NormalizedCentroid

class TransactionScope(
    private val db: SQLiteDatabase
) {

    fun getExperiments(): Collection<Experiment> {
        val query = "SELECT" +
                " ExperimentId" +
                ", Description" +
                " FROM Experiment" +
                " ORDER BY ExperimentId DESC"
        db.rawQuery(query, null).use { cursor ->
            val results = mutableListOf<Experiment>()
            while (cursor.moveToNext()) {
                results.add(Experiment(
                    experimentId = cursor.getLong(0),
                    description = cursor.getString(1),
                    trackers = emptyList()
                ))
            }
            return results
        }
    }

    fun getExperimentMeasurements(experimentId: Long): Collection<String> {
        val query = "SELECT" +
                " TrackerIndex" +
                ", Timestamp" +
                ", CentroidX" +
                ", CentroidY" +
                " FROM Measurement" +
                " WHERE ExperimentId = ?" +
                //todo group by makes sense here ???? just order?
//                " GROUP BY TrackerIndex, Timestamp" +
                " ORDER BY TrackerIndex, Timestamp"
        db.rawQuery(query, arrayOf(experimentId.toString())).use { cursor ->
            val results = mutableListOf<String>()
            while (cursor.moveToNext()) {
                results.add("${cursor.getLong(0)},${cursor.getLong(1)},${cursor.getDouble(2)},${cursor.getDouble(3)}")
            }
            return results
        }
    }

//    fun getPastData(experimentId: Long): Collection<Pair<Long, NormalizedCentroid>> {
    fun getPastData(experimentId: Long): Map<Int, Collection<Pair<Long, NormalizedCentroid>>> {
        val query = "SELECT" +
                " TrackerIndex" +
                ", Timestamp" +
                ", CentroidX" +
                ", CentroidY" +
                " FROM Measurement" +
                " WHERE ExperimentId = ?" +
//                " GROUP BY TrackerIndex, Timestamp" +
                " ORDER BY TrackerIndex, Timestamp"
        db.rawQuery(query, arrayOf(experimentId.toString())).use { cursor ->
//            val results = mutableListOf<Pair<Long, NormalizedCentroid>>()
            val results = mutableMapOf<Int, Collection<Pair<Long, NormalizedCentroid>>>()
            var currentTracker: Int? = null
            var currentMeasurements: MutableList<Pair<Long, NormalizedCentroid>>? = null


            while (cursor.moveToNext()) {
                val trackerIndex = cursor.getLong(0).toInt()

                if (trackerIndex != currentTracker) {
                    currentTracker = trackerIndex
                    currentMeasurements = mutableListOf()
                    results[trackerIndex] = currentMeasurements
                }

                currentMeasurements!!.add(cursor.getLong(1) to NormalizedCentroid(
                    centroidX = cursor.getDouble(2),
                    centroidY = cursor.getDouble(3)
                ))
//                results.add(cursor.getLong(1) to NormalizedCentroid(
//                    trackerId = cursor.getLong(0).toInt(),
//                    left = cursor.getFloat(2),
//                    right = cursor.getFloat(3),
//                    top = cursor.getFloat(4),
//                    bottom = cursor.getFloat(5)
//                )
//                )
            }
            return results
        }
    }

    fun getTrackingConfig(experimentId: Long): TrackingConfig {
        val query = "SELECT" +
                " r.ExperimentId" +
                ", r.P1X" +
                ", r.P1Y" +
                ", r.P2X" +
                ", r.P2Y" +
                ", r.Distance" +
                ", r.Unit" +
                ", roi.ExperimentId" +
                ", roi.P1X" +
                ", roi.P1Y" +
                ", roi.P2X" +
                ", roi.P2Y" +
                " FROM Experiment AS e" +
                " LEFT JOIN Ruler AS r" +
                    " ON e.ExperimentId = r.ExperimentId" +
                " LEFT JOIN Roi AS roi" +
                    " ON e.ExperimentId = roi.ExperimentId" +
                " WHERE e.ExperimentId = ?"

        db.rawQuery(query, arrayOf(experimentId.toString())).use { cursor ->
            if (cursor.moveToFirst()) {

                val ruler = if (!cursor.isNull(0)) {
                    Ruler(
                        p1 = Point2D(
                            x = cursor.getFloat(1),
                            y = cursor.getFloat(2)
                        ),
                        p2 = Point2D(
                            x = cursor.getFloat(3),
                            y = cursor.getFloat(4)
                        ),
                        realDistance = cursor.getFloat(5),
                        unit = DistanceUnit.fromSymbol(cursor.getString(6))
                    )
                } else null

                val roi = if (!cursor.isNull(7)) {
                    Roi(
                        p1 = Point2D(
                            x = cursor.getFloat(8),
                            y = cursor.getFloat(9)
                        ),
                        p2 = Point2D(
                            x = cursor.getFloat(10),
                            y = cursor.getFloat(11)
                        )
                    )
                } else null

                val queryTrackers = "SELECT" +
                        " TrackerIndex" +
                        ", Color" +
                        ", Tolerance" +
                        " FROM Tracker" +
                        " WHERE ExperimentId = ?"
                val trackers = db.rawQuery(queryTrackers, arrayOf(experimentId.toString())).use { cursorTrackers ->
                    val results = mutableListOf<ColorTrackerConfig>()
                    while (cursorTrackers.moveToNext()) {
                        results.add(ColorTrackerConfig(
                            color = cursorTrackers.getInt(0),
                            tolerance = cursorTrackers.getFloat(1)
                        ))
                    }
                    results
                }

                return TrackingConfig(
                    roi = roi,
                    ruler = ruler,
                    trackers = trackers,
                )
            } else throw DatabaseException()
        }
    }

    fun addExperiment(trackingConfig: TrackingConfig, description: String): Long {
        val values = ContentValues().apply {
            put("Description", description)
        }
        val rowId = db.insert("Experiment", null, values)
        if (rowId == -1L) throw DatabaseException()

        trackingConfig.ruler?.let { ruler ->
            values.apply {
                clear()
                put("ExperimentId", rowId)
                put("P1X", ruler.p1.x)
                put("P1Y", ruler.p1.y)
                put("P2X", ruler.p2.x)
                put("P2Y", ruler.p2.y)
                put("Distance", ruler.realDistance)
                put("Unit", ruler.unit.symbol)
            }
            val tmp = db.insert("Ruler", null, values)
            if (tmp == -1L) throw DatabaseException()
        }

        trackingConfig.roi?.let { roi ->
            values.apply {
                clear()
                put("ExperimentId", rowId)
                put("P1X", roi.p1.x)
                put("P1Y", roi.p1.y)
                put("P2X", roi.p2.x)
                put("P2Y", roi.p2.y)
            }
            val tmp = db.insert("Roi", null, values)
            if (tmp == -1L) throw DatabaseException()
        }

        trackingConfig.trackers.forEachIndexed { index, tracker ->
            values.apply {
                clear()
                put("ExperimentId", rowId)
                put("TrackerIndex", index)
                put("Color", tracker.color)
                put("Tolerance", tracker.tolerance)
            }
            val tmp = db.insert("Tracker", null, values)
            if (tmp == -1L) throw DatabaseException()
        }

        return rowId
    }

    fun addMeasurements(
        experimentId: Long,
        timestamp: Long,
        measurements: List<NormalizedBox>
    ) {
        val values = ContentValues().apply {
            put("ExperimentId", experimentId)
            put("Timestamp", timestamp)
        }
        measurements.forEach { normalizedBox ->
            values.apply {
                put("TrackerIndex", normalizedBox.trackerId)
                put("CentroidX", normalizedBox.centroidX)
                put("CentroidY", normalizedBox.centroidY)
            }
            val tmp = db.insert("Measurement", null, values)
            if (tmp == -1L) throw DatabaseException()
        }
    }
}