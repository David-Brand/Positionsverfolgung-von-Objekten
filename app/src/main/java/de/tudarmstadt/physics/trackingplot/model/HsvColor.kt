package de.tudarmstadt.physics.trackingplot.model

import kotlin.math.roundToInt

/** A colour in OpenCV's 8-bit HSV units: [h] in 0..179, [s] and [v] in 0..255. */
data class HsvColor(val h: Int, val s: Int, val v: Int) {

    /**
     * Hue is only meaningful for reasonably saturated and bright colours; grey, black and white
     * picks would match almost anything (or nothing) and are reported as not trackable.
     */
    val isTrackable: Boolean get() = s >= MIN_TRACKABLE_SATURATION && v >= MIN_TRACKABLE_VALUE

    companion object {
        const val MIN_TRACKABLE_SATURATION = 40
        const val MIN_TRACKABLE_VALUE = 40

        /** Converts an ARGB colour with the same formula OpenCV uses for COLOR_RGB2HSV. */
        fun fromArgb(argb: Int): HsvColor {
            val r = ((argb shr 16) and 0xFF) / 255f
            val g = ((argb shr 8) and 0xFF) / 255f
            val b = (argb and 0xFF) / 255f
            val max = maxOf(r, g, b)
            val min = minOf(r, g, b)
            val delta = max - min

            val hueDegrees = when {
                delta == 0f -> 0f
                max == r -> 60f * ((g - b) / delta)
                max == g -> 60f * ((b - r) / delta + 2f)
                else -> 60f * ((r - g) / delta + 4f)
            }.let { if (it < 0f) it + 360f else it }

            val saturation = if (max == 0f) 0f else delta / max
            return HsvColor(
                h = (hueDegrees / 2f).roundToInt().coerceIn(0, 179),
                s = (saturation * 255f).roundToInt().coerceIn(0, 255),
                v = (max * 255f).roundToInt().coerceIn(0, 255)
            )
        }
    }
}

/** Half-widths of the accepted HSV window around a picked colour, in OpenCV units. */
data class HsvTolerance(val hue: Int, val sat: Int, val value: Int) {
    companion object {
        /**
         * Maps the single 0..100 tolerance shown in the UI to per-channel windows. Hue gets a
         * narrow window (colours are told apart by hue), saturation and value get wide windows
         * so that shading and uneven lighting do not break the segmentation.
         */
        fun fromUiTolerance(tolerance: Int): HsvTolerance {
            val t = tolerance.coerceIn(ColorTrackerConfig.MIN_TOLERANCE, ColorTrackerConfig.MAX_TOLERANCE)
            return HsvTolerance(
                hue = (t * 0.4f).roundToInt().coerceIn(2, 90),
                sat = (t * 2.0f).roundToInt().coerceIn(10, 255),
                value = (t * 2.5f).roundToInt().coerceIn(10, 255)
            )
        }
    }
}
