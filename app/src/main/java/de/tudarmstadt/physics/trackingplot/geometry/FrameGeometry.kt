package de.tudarmstadt.physics.trackingplot.geometry

import androidx.compose.ui.geometry.Offset

/**
 * Geometry of the canonical camera frame.
 *
 * Every coordinate the app stores or plots refers to the *canonical frame*: the camera image
 * rotated to portrait orientation, cropped by CameraX to a 3:4 (width:height) viewport, and
 * normalised to 0..1 on both axes. Because the two axes are normalised by different lengths,
 * a normalised y distance is 4/3 times longer than the same normalised x distance; use
 * [HEIGHT_OVER_WIDTH] to bring y into "frame widths" before comparing or measuring lengths.
 */
object FrameGeometry {
    /** Width divided by height of the canonical (portrait) frame. */
    const val WIDTH_OVER_HEIGHT = 3f / 4f

    /** Height divided by width of the canonical (portrait) frame. */
    const val HEIGHT_OVER_WIDTH = 4f / 3f

    /** Aspect ratio as integers for CameraX's Rational. */
    const val ASPECT_WIDTH = 3
    const val ASPECT_HEIGHT = 4
}

/**
 * Rotation (clockwise degrees) that maps the canonical frame onto what the viewfinder shows for
 * the current display orientation. In portrait the two coincide; in landscape the preview is the
 * canonical frame turned by 90 or 270 degrees.
 */
enum class FrameRotation(val degrees: Int) {
    ROTATION_0(0),
    ROTATION_90(90),
    ROTATION_180(180),
    ROTATION_270(270);

    /** True when the displayed frame has the canonical width and height swapped. */
    val swapsAxes: Boolean get() = degrees == 90 || degrees == 270

    /** Width divided by height of the frame as displayed. */
    val displayWidthOverHeight: Float
        get() = if (swapsAxes) FrameGeometry.HEIGHT_OVER_WIDTH else FrameGeometry.WIDTH_OVER_HEIGHT

    /** Canonical normalised point -> displayed normalised point. */
    fun toDisplay(p: Offset): Offset = when (this) {
        ROTATION_0 -> p
        ROTATION_90 -> Offset(1f - p.y, p.x)
        ROTATION_180 -> Offset(1f - p.x, 1f - p.y)
        ROTATION_270 -> Offset(p.y, 1f - p.x)
    }

    /** Displayed normalised point -> canonical normalised point. */
    fun fromDisplay(p: Offset): Offset = when (this) {
        ROTATION_0 -> p
        ROTATION_90 -> Offset(p.y, 1f - p.x)
        ROTATION_180 -> Offset(1f - p.x, 1f - p.y)
        ROTATION_270 -> Offset(1f - p.y, p.x)
    }

    companion object {
        /**
         * @param surfaceRotation one of android.view.Surface.ROTATION_0/90/180/270 (0..3),
         *   i.e. the current display rotation.
         */
        fun fromDisplayRotation(surfaceRotation: Int): FrameRotation {
            val displayDegrees = (surfaceRotation.coerceIn(0, 3)) * 90
            return when ((360 - displayDegrees) % 360) {
                90 -> ROTATION_90
                180 -> ROTATION_180
                270 -> ROTATION_270
                else -> ROTATION_0
            }
        }
    }
}
