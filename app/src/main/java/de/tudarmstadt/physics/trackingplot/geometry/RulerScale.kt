package de.tudarmstadt.physics.trackingplot.geometry

import de.tudarmstadt.physics.trackingplot.model.DistanceUnit
import de.tudarmstadt.physics.trackingplot.model.Point2D
import de.tudarmstadt.physics.trackingplot.model.Ruler
import kotlin.math.sqrt

/**
 * Converts normalised canonical coordinates into real-world lengths using a [Ruler].
 *
 * The ruler length is measured in an isotropic space (y scaled by [FrameGeometry.HEIGHT_OVER_WIDTH]
 * so that both axes are in units of "frame widths"); the resulting scale therefore holds for
 * rulers of any orientation and is applied consistently to both axes.
 */
class RulerScale private constructor(
    /** Real-world length of the full canonical frame width. */
    val unitsPerFrameWidth: Double,
    val unit: DistanceUnit
) {
    fun x(normalizedX: Double): Double = normalizedX * unitsPerFrameWidth

    fun y(normalizedY: Double): Double = normalizedY * FrameGeometry.HEIGHT_OVER_WIDTH * unitsPerFrameWidth

    companion object {
        /** Rulers shorter than this fraction of the frame width are rejected as degenerate. */
        const val MIN_NORMALIZED_LENGTH = 0.02f

        /** Length of the segment p1-p2 in units of the canonical frame width. */
        fun normalizedLength(p1: Point2D, p2: Point2D): Float {
            val dx = p2.x - p1.x
            val dy = (p2.y - p1.y) * FrameGeometry.HEIGHT_OVER_WIDTH
            return sqrt(dx * dx + dy * dy)
        }

        fun isValid(ruler: Ruler): Boolean =
            ruler.realDistance.isFinite() && ruler.realDistance > 0f &&
                normalizedLength(ruler.p1, ruler.p2) >= MIN_NORMALIZED_LENGTH

        /** Returns null for a missing or degenerate ruler. */
        fun from(ruler: Ruler?): RulerScale? {
            if (ruler == null || !isValid(ruler)) return null
            val length = normalizedLength(ruler.p1, ruler.p2)
            return RulerScale(ruler.realDistance.toDouble() / length, ruler.unit)
        }
    }
}
