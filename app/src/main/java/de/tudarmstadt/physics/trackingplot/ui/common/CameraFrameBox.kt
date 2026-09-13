package de.tudarmstadt.physics.trackingplot.ui.common

import android.view.Surface
import androidx.camera.compose.CameraXViewfinder
import androidx.camera.core.SurfaceRequest
import androidx.camera.viewfinder.core.ImplementationMode
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.isSpecified
import de.tudarmstadt.physics.trackingplot.geometry.FrameRotation
import de.tudarmstadt.physics.trackingplot.model.Point2D

/**
 * Maps between canonical frame coordinates (normalised, portrait) and pixel positions inside the
 * viewfinder box on screen for the current display rotation.
 */
@Immutable
class FrameMapper(val rotation: FrameRotation, val boxSize: Size) {

    /** Canonical normalised point -> pixel offset inside the box. */
    fun toBox(p: Point2D): Offset = toBox(Offset(p.x, p.y))

    fun toBox(canonical: Offset): Offset {
        val d = rotation.toDisplay(canonical)
        return Offset(d.x * boxSize.width, d.y * boxSize.height)
    }

    /** Pixel offset inside the box -> canonical normalised point, clamped to the frame. */
    fun toCanonical(box: Offset): Point2D {
        val display = Offset(
            (box.x / boxSize.width).coerceIn(0f, 1f),
            (box.y / boxSize.height).coerceIn(0f, 1f)
        )
        val c = rotation.fromDisplay(display)
        return Point2D(c.x.coerceIn(0f, 1f), c.y.coerceIn(0f, 1f))
    }

    /** Axis-aligned pixel rectangle of a canonical rectangle given by two opposite corners. */
    fun toBoxRect(a: Point2D, b: Point2D): Rect {
        val pa = toBox(a)
        val pb = toBox(b)
        return Rect(
            left = minOf(pa.x, pb.x),
            top = minOf(pa.y, pb.y),
            right = maxOf(pa.x, pb.x),
            bottom = maxOf(pa.y, pb.y)
        )
    }
}

/** The rotation that maps the canonical frame onto the preview for the current display rotation. */
@Composable
fun rememberFrameRotation(): FrameRotation {
    val view = LocalView.current
    val configuration = LocalConfiguration.current
    return remember(configuration, view) {
        val rotation = try {
            view.display?.rotation ?: Surface.ROTATION_0
        } catch (e: Exception) {
            Surface.ROTATION_0
        }
        FrameRotation.fromDisplayRotation(rotation)
    }
}

/**
 * Shows the camera preview in the largest box with the frame's display aspect ratio that fits the
 * incoming constraints, and lets [overlay] draw on top of it using a [FrameMapper] whose box size
 * is in pixels.
 *
 * @param maxBoxWidth optional cap for the box width (e.g. to leave room for other content).
 * @param viewfinderModifier modifier applied to the viewfinder itself (e.g. gesture handling).
 */
@Composable
fun CameraFrameBox(
    surfaceRequest: SurfaceRequest?,
    modifier: Modifier = Modifier,
    rotation: FrameRotation = rememberFrameRotation(),
    contentAlignment: Alignment = Alignment.Center,
    maxBoxWidth: Dp = Dp.Unspecified,
    viewfinderModifier: Modifier = Modifier,
    overlay: @Composable BoxScope.(FrameMapper) -> Unit
) {
    BoxWithConstraints(
        modifier = modifier,
        contentAlignment = contentAlignment
    ) {
        val aspect = rotation.displayWidthOverHeight
        val availableWidth = if (maxBoxWidth.isSpecified) minOf(maxWidth, maxBoxWidth) else maxWidth
        val availableHeight = maxHeight
        val useFullWidth = availableWidth / availableHeight <= aspect
        val boxWidth = if (useFullWidth) availableWidth else availableHeight * aspect
        val boxHeight = if (useFullWidth) availableWidth / aspect else availableHeight

        val density = LocalDensity.current
        val mapper = remember(rotation, boxWidth, boxHeight, density) {
            FrameMapper(rotation, with(density) { Size(boxWidth.toPx(), boxHeight.toPx()) })
        }

        Box(
            modifier = Modifier
                .width(boxWidth)
                .height(boxHeight)
                .background(Color.Black)
        ) {
            if (surfaceRequest != null) {
                CameraXViewfinder(
                    surfaceRequest = surfaceRequest,
                    implementationMode = ImplementationMode.EXTERNAL,
                    contentScale = ContentScale.Fit,
                    modifier = Modifier
                        .fillMaxSize()
                        .then(viewfinderModifier)
                )
            }
            overlay(mapper)
        }
    }
}
