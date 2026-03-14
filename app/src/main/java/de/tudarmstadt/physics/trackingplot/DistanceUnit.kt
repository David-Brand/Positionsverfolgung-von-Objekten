package de.tudarmstadt.physics.trackingplot

enum class DistanceUnit(val symbol: String, val multiplierToMeters: Float) {
    METER("m", 1f),
    CENTIMETER("cm", 0.01f),
    MILLIMETER("mm", 0.001f);

    companion object {
        fun fromSymbol(symbol: String): DistanceUnit {
            return requireNotNull(entries.find { it.symbol == symbol })
        }
    }
}
