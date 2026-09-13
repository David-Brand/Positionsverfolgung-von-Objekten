package de.tudarmstadt.physics.trackingplot.geometry

import de.tudarmstadt.physics.trackingplot.model.DistanceUnit
import de.tudarmstadt.physics.trackingplot.model.Point2D
import de.tudarmstadt.physics.trackingplot.model.Ruler
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class RulerScaleTest {

    @Test
    fun horizontalRulerScalesXDirectly() {
        // A 10 cm ruler spanning half the frame width: the full width is 20 cm.
        val ruler = Ruler(Point2D(0.25f, 0.5f), Point2D(0.75f, 0.5f), 10f, DistanceUnit.CENTIMETER)
        val scale = RulerScale.from(ruler)
        assertNotNull(scale)
        assertEquals(20.0, scale!!.unitsPerFrameWidth, 1e-6)
        assertEquals(20.0, scale.x(1.0), 1e-6)
        assertEquals(5.0, scale.x(0.25), 1e-6)
    }

    @Test
    fun verticalRulerAccountsForAspectRatio() {
        // Vertical ruler spanning half the frame height. The frame height is 4/3 of the width,
        // so half the height is 2/3 of a width: 10 cm = 2/3 width -> width = 15 cm.
        val ruler = Ruler(Point2D(0.5f, 0.25f), Point2D(0.5f, 0.75f), 10f, DistanceUnit.CENTIMETER)
        val scale = RulerScale.from(ruler)!!
        assertEquals(15.0, scale.unitsPerFrameWidth, 1e-4)
        // The full frame height is 4/3 * 15 cm = 20 cm.
        assertEquals(20.0, scale.y(1.0), 1e-4)
        // Both axes agree on physical length: 0.3 of the width equals 0.225 of the height.
        assertEquals(scale.x(0.3), scale.y(0.225), 1e-6)
    }

    @Test
    fun diagonalRulerUsesIsotropicLength() {
        // dx = 0.3 widths, dy = 0.3 heights = 0.4 widths -> length 0.5 widths.
        val ruler = Ruler(Point2D(0.1f, 0.1f), Point2D(0.4f, 0.4f), 5f, DistanceUnit.METER)
        val scale = RulerScale.from(ruler)!!
        assertEquals(10.0, scale.unitsPerFrameWidth, 1e-4)
    }

    @Test
    fun degenerateRulersAreRejected() {
        assertNull(RulerScale.from(null))
        assertNull(RulerScale.from(Ruler(Point2D(0.5f, 0.5f), Point2D(0.5f, 0.5f), 10f, DistanceUnit.METER)))
        assertNull(RulerScale.from(Ruler(Point2D(0.1f, 0.1f), Point2D(0.9f, 0.9f), 0f, DistanceUnit.METER)))
        assertNull(RulerScale.from(Ruler(Point2D(0.1f, 0.1f), Point2D(0.9f, 0.9f), -1f, DistanceUnit.METER)))
        assertNull(RulerScale.from(Ruler(Point2D(0.1f, 0.1f), Point2D(0.9f, 0.9f), Float.NaN, DistanceUnit.METER)))
    }
}
