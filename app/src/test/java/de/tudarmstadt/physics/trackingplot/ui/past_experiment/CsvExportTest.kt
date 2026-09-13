package de.tudarmstadt.physics.trackingplot.ui.past_experiment

import de.tudarmstadt.physics.trackingplot.model.ColorTrackerConfig
import de.tudarmstadt.physics.trackingplot.model.DistanceUnit
import de.tudarmstadt.physics.trackingplot.model.MeasurementSample
import de.tudarmstadt.physics.trackingplot.model.Point2D
import de.tudarmstadt.physics.trackingplot.model.Ruler
import de.tudarmstadt.physics.trackingplot.model.TrackingConfig
import org.junit.Assert.assertEquals
import org.junit.Test

class CsvExportTest {

    private val samples = mapOf(
        1 to listOf(MeasurementSample(1, 20, 0.5, 0.5)),
        0 to listOf(MeasurementSample(0, 0, 0.25, 0.75), MeasurementSample(0, 100, 0.5, 0.25))
    )

    @Test
    fun withoutRulerOnlyNormalisedColumnsAreWritten() {
        val config = TrackingConfig(null, null, listOf(ColorTrackerConfig(0, 40)), 10, 1f)
        val csv = PastExperimentViewModel.buildCsv(config, samples)
        assertEquals(
            "trackerIndex,timestamp,centroidX,centroidY\n" +
                "0,0,0.25,0.75\n" +
                "0,100,0.5,0.25\n" +
                "1,20,0.5,0.5\n",
            csv
        )
    }

    @Test
    fun withRulerRealWorldColumnsAreAppended() {
        // Horizontal ruler over half the width = 10 cm -> width 20 cm, height 26.67 cm.
        val ruler = Ruler(Point2D(0.25f, 0.5f), Point2D(0.75f, 0.5f), 10f, DistanceUnit.CENTIMETER)
        val config = TrackingConfig(null, ruler, listOf(ColorTrackerConfig(0, 40)), 10, 1f)
        val lines = PastExperimentViewModel.buildCsv(config, samples).lines()
        assertEquals("trackerIndex,timestamp,centroidX,centroidY,x_cm,y_cm", lines[0])
        val first = lines[1].split(',')
        assertEquals(6, first.size)
        assertEquals(5.0, first[4].toDouble(), 1e-6)
        assertEquals(20.0, first[5].toDouble(), 1e-6)
    }
}
