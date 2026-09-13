package de.tudarmstadt.physics.trackingplot.model

enum class DistanceUnit(val symbol: String, val multiplierToMeters: Float) {
    METER("m", 1f),
    CENTIMETER("cm", 0.01f),
    MILLIMETER("mm", 0.001f);

    companion object {
        fun fromSymbolOrNull(symbol: String): DistanceUnit? = entries.find { it.symbol == symbol }
    }
}
