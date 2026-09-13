package de.tudarmstadt.physics.trackingplot.geometry

import android.view.Surface
import androidx.compose.ui.geometry.Offset
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FrameRotationTest {

    private val eps = 1e-6f

    private fun assertOffset(expected: Offset, actual: Offset) {
        assertEquals("x", expected.x, actual.x, eps)
        assertEquals("y", expected.y, actual.y, eps)
    }

    @Test
    fun portraitIsIdentity() {
        val rotation = FrameRotation.fromDisplayRotation(Surface.ROTATION_0)
        assertEquals(FrameRotation.ROTATION_0, rotation)
        assertFalse(rotation.swapsAxes)
        assertOffset(Offset(0.2f, 0.7f), rotation.toDisplay(Offset(0.2f, 0.7f)))
        assertEquals(FrameGeometry.WIDTH_OVER_HEIGHT, rotation.displayWidthOverHeight, eps)
    }

    @Test
    fun landscapeRotation90MapsCornersCounterClockwise() {
        // Display rotated by 90 degrees: the canonical frame appears turned 90 degrees CCW (=270 CW).
        val rotation = FrameRotation.fromDisplayRotation(Surface.ROTATION_90)
        assertEquals(FrameRotation.ROTATION_270, rotation)
        assertTrue(rotation.swapsAxes)
        assertOffset(Offset(0f, 1f), rotation.toDisplay(Offset(0f, 0f))) // top-left -> bottom-left
        assertOffset(Offset(0f, 0f), rotation.toDisplay(Offset(1f, 0f))) // top-right -> top-left
        assertOffset(Offset(1f, 0f), rotation.toDisplay(Offset(1f, 1f))) // bottom-right -> top-right
        assertEquals(FrameGeometry.HEIGHT_OVER_WIDTH, rotation.displayWidthOverHeight, eps)
    }

    @Test
    fun landscapeRotation270MapsCornersClockwise() {
        val rotation = FrameRotation.fromDisplayRotation(Surface.ROTATION_270)
        assertEquals(FrameRotation.ROTATION_90, rotation)
        assertOffset(Offset(1f, 0f), rotation.toDisplay(Offset(0f, 0f))) // top-left -> top-right
        assertOffset(Offset(0f, 0f), rotation.toDisplay(Offset(0f, 1f))) // bottom-left -> top-left
    }

    @Test
    fun fromDisplayIsInverseOfToDisplay() {
        val points = listOf(Offset(0f, 0f), Offset(1f, 1f), Offset(0.25f, 0.8f), Offset(0.9f, 0.1f))
        for (rotation in FrameRotation.entries) {
            for (p in points) {
                assertOffset(p, rotation.fromDisplay(rotation.toDisplay(p)))
                assertOffset(p, rotation.toDisplay(rotation.fromDisplay(p)))
            }
        }
    }
}
