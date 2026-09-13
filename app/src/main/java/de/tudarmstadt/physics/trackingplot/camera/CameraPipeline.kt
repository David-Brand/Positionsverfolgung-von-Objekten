package de.tudarmstadt.physics.trackingplot.camera

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.util.Rational
import android.util.Size
import android.view.Surface
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceRequest
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.lifecycle.awaitInstance
import androidx.lifecycle.LifecycleOwner
import de.tudarmstadt.physics.trackingplot.geometry.FrameGeometry
import de.tudarmstadt.physics.trackingplot.model.FrameResult
import de.tudarmstadt.physics.trackingplot.model.Roi
import de.tudarmstadt.physics.trackingplot.model.TrackedObject
import de.tudarmstadt.physics.trackingplot.tracker.NativeTracker
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withTimeout
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference

/**
 * Owns the CameraX use cases (Preview + ImageAnalysis) for one screen flow and runs the colour
 * tracker on every analysed frame.
 *
 * The use cases are created once and can be re-bound to a new lifecycle owner (e.g. after a
 * rotation) with [bind]. ImageAnalysis is configured with a fixed target rotation so that frames
 * are always converted into the canonical portrait frame, independent of the display orientation;
 * the preview follows the display and the UI maps between the two with FrameRotation.
 */
class CameraPipeline(
    private val tracker: NativeTracker,
    /** Queried on the analysis thread for every frame; false skips tracking (frames are dropped). */
    private val isAnalysisEnabled: () -> Boolean
) : AutoCloseable {

    sealed interface BindState {
        data object Unbound : BindState
        data object Bound : BindState
        data class Failed(val error: Throwable) : BindState
    }

    private val analysisExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "camera-analysis")
    }
    private val processor = FrameProcessor()

    private val _surfaceRequests = MutableStateFlow<SurfaceRequest?>(null)
    val surfaceRequests: StateFlow<SurfaceRequest?> = _surfaceRequests

    // Only the newest result is kept when the UI falls behind, so the overlay never shows
    // positions that are several frames old.
    private val _frames = MutableSharedFlow<FrameResult>(
        extraBufferCapacity = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    /** Tracking results, emitted on the analysis thread. */
    val frames: SharedFlow<FrameResult> = _frames

    private val _bindState = MutableStateFlow<BindState>(BindState.Unbound)
    val bindState: StateFlow<BindState> = _bindState

    private val _zoomRatio = MutableStateFlow(1f)

    /** Zoom ratio that is (or will be, after binding) applied to the camera. */
    val zoomRatio: StateFlow<Float> = _zoomRatio

    private val _zoomRange = MutableStateFlow(1f..1f)
    val zoomRange: StateFlow<ClosedFloatingPointRange<Float>> = _zoomRange

    /** Region of interest in canonical coordinates; null analyses the whole frame. */
    @Volatile
    var roi: Roi? = null

    @Volatile
    private var closed = false

    @Volatile
    private var camera: Camera? = null

    private val snapshotRequest = AtomicReference<CompletableDeferred<Bitmap>?>(null)

    private val previewSelector = ResolutionSelector.Builder()
        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
        .build()

    // Analysis runs at (or near) 640x480 like CameraX's default: without an explicit strategy a
    // custom ResolutionSelector would pick the highest available 4:3 size, which makes the colour
    // conversion and tracking many times slower and the overlay lag behind the preview.
    private val analysisSelector = ResolutionSelector.Builder()
        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
        .setResolutionStrategy(
            ResolutionStrategy(ANALYSIS_TARGET_SIZE, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
        )
        .build()

    private val preview: Preview = Preview.Builder()
        .setResolutionSelector(previewSelector)
        .build()
        .apply {
            setSurfaceProvider { request -> _surfaceRequests.value = request }
        }

    private val imageAnalysis: ImageAnalysis = ImageAnalysis.Builder()
        .setResolutionSelector(analysisSelector)
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        // Fixed target rotation: rotationDegrees is then the constant sensor orientation and
        // FrameProcessor always produces the same (portrait) frame.
        .setTargetRotation(Surface.ROTATION_0)
        .build()
        .apply {
            setAnalyzer(analysisExecutor, ::analyze)
        }

    private val useCaseGroup: UseCaseGroup = UseCaseGroup.Builder()
        .setViewPort(
            ViewPort.Builder(
                Rational(FrameGeometry.ASPECT_WIDTH, FrameGeometry.ASPECT_HEIGHT),
                Surface.ROTATION_0
            ).build()
        )
        .addUseCase(preview)
        .addUseCase(imageAnalysis)
        .build()

    /**
     * Binds the use cases to [lifecycleOwner] and keeps them bound until the calling coroutine is
     * cancelled. Binding failures (unsupported camera configuration, no camera) are reported via
     * [bindState] instead of being thrown.
     */
    suspend fun bind(context: Context, lifecycleOwner: LifecycleOwner) {
        val provider = ProcessCameraProvider.awaitInstance(context.applicationContext)
        val generation = synchronized(bindLock) { ++bindGeneration }
        var boundCamera: Camera? = null
        try {
            provider.unbindAll()
            val cam = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, useCaseGroup)
            boundCamera = cam
            camera = cam
            cam.cameraInfo.zoomState.value?.let { state ->
                _zoomRange.value = state.minZoomRatio..state.maxZoomRatio
            }
            applyZoom(cam, _zoomRatio.value)
            _bindState.value = BindState.Bound
            awaitCancellation()
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Binding the camera failed", e)
            _bindState.value = BindState.Failed(e)
        } finally {
            // A newer bind of this pipeline (the next setup screen binds while this one is still
            // being torn down) already owns [camera]; never clear or unbind what it set up.
            if (camera === boundCamera) camera = null
            val stillCurrent = synchronized(bindLock) { bindGeneration == generation }
            if (stillCurrent) {
                provider.unbindAll()
                if (_bindState.value == BindState.Bound) _bindState.value = BindState.Unbound
            }
        }
    }

    fun setZoomRatio(ratio: Float) {
        val range = _zoomRange.value
        val clamped = ratio.coerceIn(range.start, range.endInclusive)
        _zoomRatio.value = clamped
        camera?.let { applyZoom(it, clamped) }
    }

    private fun applyZoom(camera: Camera, ratio: Float) {
        val range = _zoomRange.value
        val clamped = ratio.coerceIn(range.start, range.endInclusive)
        if (clamped != ratio) _zoomRatio.value = clamped
        camera.cameraControl.setZoomRatio(clamped)
    }

    /**
     * Captures the next analysed frame as a bitmap in canonical orientation. This is the exact
     * image the tracker sees, which makes it the right source for picking colours.
     */
    suspend fun captureSnapshot(timeoutMs: Long = 3_000L): Bitmap {
        val deferred = CompletableDeferred<Bitmap>()
        snapshotRequest.getAndSet(deferred)?.cancel()
        try {
            return withTimeout(timeoutMs) { deferred.await() }
        } finally {
            snapshotRequest.compareAndSet(deferred, null)
        }
    }

    private fun analyze(image: ImageProxy) {
        try {
            if (closed) return
            val snapshot = snapshotRequest.getAndSet(null)
            val analysisWanted = isAnalysisEnabled()
            if (snapshot == null && !analysisWanted) return

            val frame = processor.toCanonicalFrame(image)
            if (frame == null) {
                snapshot?.completeExceptionally(IllegalStateException("Unsupported frame format ${image.format}"))
                return
            }
            snapshot?.let { request ->
                try {
                    request.complete(processor.toBitmap(frame))
                } catch (e: Exception) {
                    request.completeExceptionally(e)
                }
            }
            if (!analysisWanted) return

            val width = frame.cols()
            val height = frame.rows()
            val results = tracker.update(frame, roi?.toPixelRect(width, height))
            val objects = results.mapNotNull { r ->
                if (!r.found) return@mapNotNull null
                TrackedObject(
                    trackerId = r.trackerId,
                    left = r.x.toFloat() / width,
                    top = r.y.toFloat() / height,
                    right = (r.x + r.width).toFloat() / width,
                    bottom = (r.y + r.height).toFloat() / height,
                    centroidX = r.centroidX / width,
                    centroidY = r.centroidY / height
                )
            }
            _frames.tryEmit(FrameResult(image.imageInfo.timestamp, objects))
        } catch (e: Exception) {
            // Never let a single bad frame take the process down.
            Log.e(TAG, "Frame analysis failed", e)
        } finally {
            image.close()
        }
    }

    override fun close() {
        closed = true
        imageAnalysis.clearAnalyzer()
        snapshotRequest.getAndSet(null)?.cancel()
        // Runs after any frame that is currently being analysed on the single analysis thread.
        analysisExecutor.execute { processor.close() }
        analysisExecutor.shutdown()
    }

    private companion object {
        const val TAG = "CameraPipeline"
        val ANALYSIS_TARGET_SIZE = Size(640, 480)

        /** Guards against a cancelled bind unbinding a newer one (possibly of another pipeline). */
        val bindLock = Any()
        var bindGeneration = 0L
    }
}
