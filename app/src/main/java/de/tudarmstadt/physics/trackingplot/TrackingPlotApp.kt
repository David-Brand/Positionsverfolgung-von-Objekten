package de.tudarmstadt.physics.trackingplot

import android.app.Application

class TrackingPlotApp: Application() {

    companion object {
        lateinit var trackingModule: TrackingModule
    }

    override fun onCreate() {
        super.onCreate()

        trackingModule = TrackingModule(this)
    }
}