package de.tudarmstadt.physics.trackingplot.ui.common

import androidx.compose.ui.graphics.Color

/** Display colour and name of each tracker slot (by index) used consistently across all screens. */
object TrackerColors {
    private val colors = listOf(
        Color(0xFFE53935), // red
        Color(0xFF43A047), // green
        Color(0xFF1E88E5)  // blue
    )

    fun color(trackerIndex: Int): Color = colors.getOrElse(trackerIndex) { Color.Gray }

    /** 1-based human readable name, e.g. "Tracker 2". */
    fun name(trackerIndex: Int): String = "Tracker ${trackerIndex + 1}"
}
