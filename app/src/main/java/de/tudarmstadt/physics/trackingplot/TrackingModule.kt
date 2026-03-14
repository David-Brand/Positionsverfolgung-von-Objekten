package de.tudarmstadt.physics.trackingplot

import android.app.Application
import de.tudarmstadt.physics.trackingplot.tracker2.TrackingSession
import org.opencv.android.OpenCVLoader

class TrackingModule(
    val app: Application
) {

    init {
        OpenCVLoader.initLocal()

        System.loadLibrary("native-lib")
    }

    val trackingSession = TrackingSession
}