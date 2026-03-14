package de.tudarmstadt.physics.trackingplot.db

class DatabaseException: Exception {
    constructor(cause: Throwable): super(cause)

    constructor(): super()
}