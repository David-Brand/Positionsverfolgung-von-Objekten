package de.tudarmstadt.physics.trackingplot

import android.app.Application
import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import org.opencv.android.OpenCVLoader

class TrackingPlotApp : Application() {

    /** Scope for work that must outlive a screen, e.g. flushing measurements to the database. */
    val appScope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    /** Non-null when the native libraries could not be loaded; the UI shows an error instead of crashing. */
    var nativeLibraryError: Throwable? = null
        private set

    override fun onCreate() {
        super.onCreate()
        nativeLibraryError = loadNativeLibraries()
    }

    private fun loadNativeLibraries(): Throwable? {
        return try {
            check(OpenCVLoader.initLocal()) { "OpenCV could not be initialised" }
            System.loadLibrary("native-lib")
            null
        } catch (e: Throwable) {
            Log.e(TAG, "Native libraries failed to load", e)
            e
        }
    }

    companion object {
        private const val TAG = "TrackingPlotApp"

        fun from(context: Context): TrackingPlotApp = context.applicationContext as TrackingPlotApp
    }
}
