package de.tudarmstadt.physics.trackingplot.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class HsvColorTest {

    @Test
    fun primaryColoursMapToOpenCvHues() {
        assertEquals(HsvColor(0, 255, 255), HsvColor.fromArgb(0xFFFF0000.toInt()))
        assertEquals(HsvColor(60, 255, 255), HsvColor.fromArgb(0xFF00FF00.toInt()))
        assertEquals(HsvColor(120, 255, 255), HsvColor.fromArgb(0xFF0000FF.toInt()))
        assertEquals(HsvColor(30, 255, 255), HsvColor.fromArgb(0xFFFFFF00.toInt())) // yellow
    }

    @Test
    fun greyHasNoSaturation() {
        val grey = HsvColor.fromArgb(0xFF808080.toInt())
        assertEquals(0, grey.s)
        assertEquals(128, grey.v)
        assertFalse(grey.isTrackable)
        assertFalse(HsvColor.fromArgb(0xFF000000.toInt()).isTrackable)
        assertTrue(HsvColor.fromArgb(0xFFFF0000.toInt()).isTrackable)
    }

    @Test
    fun darkRedKeepsItsHue() {
        val darkRed = HsvColor.fromArgb(0xFF800000.toInt())
        assertEquals(0, darkRed.h)
        assertEquals(255, darkRed.s)
        assertEquals(128, darkRed.v)
    }

    @Test
    fun uiToleranceMapsToNarrowHueAndWideSaturationValueWindows() {
        val t = HsvTolerance.fromUiTolerance(ColorTrackerConfig.DEFAULT_TOLERANCE)
        assertEquals(16, t.hue)
        assertEquals(80, t.sat)
        assertEquals(100, t.value)

        val min = HsvTolerance.fromUiTolerance(0)
        assertTrue(min.hue >= 2 && min.sat >= 10 && min.value >= 10)

        val max = HsvTolerance.fromUiTolerance(1000)
        assertEquals(40, max.hue)
        assertEquals(200, max.sat)
        assertEquals(250, max.value)
    }
}
