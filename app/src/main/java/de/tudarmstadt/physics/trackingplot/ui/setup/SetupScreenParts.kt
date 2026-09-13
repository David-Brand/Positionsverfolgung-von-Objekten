package de.tudarmstadt.physics.trackingplot.ui.setup

import android.content.res.Configuration
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.PointerInputScope
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChange
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import de.tudarmstadt.physics.trackingplot.R
import de.tudarmstadt.physics.trackingplot.camera.CameraPipeline
import de.tudarmstadt.physics.trackingplot.model.Point2D
import de.tudarmstadt.physics.trackingplot.ui.common.CameraFrameBox
import de.tudarmstadt.physics.trackingplot.ui.common.FrameMapper
import kotlin.math.roundToInt

/** Colour used to draw the ruler reference line on the camera overlay. */
internal val SetupRulerColor = Color(0xFF00E5FF)

/** Colour used to draw the region-of-interest rectangle on the camera overlay. */
internal val SetupRoiColor = Color(0xFF43A047)

/** Diameter of a draggable point handle. */
private val HandleSize = 32.dp

/** Diameter of a draggable point handle while it is being dragged. */
private val HandleSizeDragged = 38.dp

/** Width of the crosshair drawn through the centre of a drag handle. */
private val HandleCrosshairWidth = 0.75.dp

/** Colour of the dark outline drawn under the white border/crosshair so both stay visible on bright backgrounds. */
private val HandleShadowColor = Color.Black.copy(alpha = 0.35f)

/** Width of the dark line drawn under each white crosshair line. */
private val HandleCrosshairShadowWidth = 2.dp

/** Width of the dark ring drawn just outside the white border. */
private val HandleBorderShadowWidth = 1.dp

/** Points closer than this to a drag start are considered "picked up". */
private val HandleTouchSlop = 40.dp

/** Height of the fade drawn over the bottom edge of a scrollable controls area. */
private val ScrollFadeHeight = 24.dp

private const val TOTAL_SETUP_STEPS = 3

/**
 * Compact single-line header shared by all three setup screens: the step title at the start and a
 * "Step n of 3" indicator at the end of the same row, so the camera box below gets as much space as
 * possible.
 */
@Composable
internal fun SetupStepHeader(step: Int, title: String, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = title, style = MaterialTheme.typography.titleMedium)
        Text(
            text = stringResource(R.string.setup_step_indicator, step, TOTAL_SETUP_STEPS),
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

/**
 * Shared scaffold for the three setup screens: step header on top, then camera box + scrollable
 * controls arranged for the current orientation (camera on top in portrait, camera on the left in
 * landscape), and finally [bottomBar] pinned below the scrollable area (portrait: below the
 * controls; landscape: at the bottom of the right-hand controls column) so its actions are always
 * reachable without scrolling.
 */
@Composable
internal fun SetupScreenScaffold(
    step: Int,
    title: String,
    setupViewModel: SetupViewModel,
    cameraOverlay: @Composable BoxScope.(FrameMapper) -> Unit,
    controls: @Composable ColumnScope.() -> Unit,
    bottomBar: @Composable () -> Unit
) {
    val configuration = LocalConfiguration.current
    val isLandscape = configuration.orientation == Configuration.ORIENTATION_LANDSCAPE
    Column(modifier = Modifier.fillMaxSize().safeDrawingPadding()) {
        SetupStepHeader(step = step, title = title)
        if (isLandscape) {
            Row(modifier = Modifier.fillMaxSize()) {
                SetupCameraBox(
                    setupViewModel = setupViewModel,
                    modifier = Modifier.weight(1.2f).fillMaxHeight(),
                    overlay = cameraOverlay
                )
                Column(modifier = Modifier.weight(1f).fillMaxHeight()) {
                    SetupScrollableControls(
                        modifier = Modifier.weight(1f).fillMaxWidth(),
                        fillAvailableHeight = true,
                        controls = controls
                    )
                    SetupBottomBar(bottomBar)
                }
            }
        } else {
            // The camera gets all remaining vertical space; the controls area only takes as much
            // height as it needs, up to 40% of the screen height, and scrolls beyond that.
            val maxControlsHeight = (configuration.screenHeightDp * 0.4f).dp
            Column(modifier = Modifier.fillMaxSize()) {
                SetupCameraBox(
                    setupViewModel = setupViewModel,
                    modifier = Modifier.weight(1f).fillMaxWidth(),
                    overlay = cameraOverlay
                )
                SetupScrollableControls(
                    modifier = Modifier.heightIn(max = maxControlsHeight).fillMaxWidth(),
                    fillAvailableHeight = false,
                    controls = controls
                )
                SetupBottomBar(bottomBar)
            }
        }
    }
}

/**
 * The scrollable controls area of a setup screen: [controls] in a vertically scrolling column,
 * with a bottom fade shown while there is more content below the visible area. When
 * [fillAvailableHeight] is false the column only takes as much height as its content needs (up to
 * whatever max height [modifier] constrains it to) instead of always filling the space given to it.
 */
@Composable
private fun SetupScrollableControls(
    modifier: Modifier = Modifier,
    fillAvailableHeight: Boolean,
    controls: @Composable ColumnScope.() -> Unit
) {
    val scrollState = rememberScrollState()
    Box(modifier = modifier) {
        Column(
            modifier = (if (fillAvailableHeight) Modifier.fillMaxSize() else Modifier.fillMaxWidth())
                .verticalScroll(scrollState)
                .padding(horizontal = 12.dp, vertical = 6.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
            content = controls
        )
        if (scrollState.canScrollForward) {
            Box(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .height(ScrollFadeHeight)
                    .background(
                        Brush.verticalGradient(
                            colors = listOf(Color.Transparent, MaterialTheme.colorScheme.background)
                        )
                    )
            )
        }
    }
}

/** Divider + always-visible action row pinned below the scrollable controls of a setup screen. */
@Composable
private fun SetupBottomBar(bottomBar: @Composable () -> Unit) {
    HorizontalDivider()
    Box(modifier = Modifier.fillMaxWidth().padding(16.dp)) {
        bottomBar()
    }
}

/**
 * Binds the camera pipeline, shows the viewfinder (or a loading/failure state), and wires up
 * pinch-to-zoom together with a zoom indicator. [overlay] draws on top using the [FrameMapper] for
 * the current box size and display rotation.
 */
@Composable
private fun SetupCameraBox(
    setupViewModel: SetupViewModel,
    modifier: Modifier = Modifier,
    maxBoxWidth: Dp = Dp.Unspecified,
    overlay: @Composable BoxScope.(FrameMapper) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var retryToken by rememberSaveable { mutableIntStateOf(0) }
    LaunchedEffect(lifecycleOwner, retryToken) {
        setupViewModel.pipeline.bind(context, lifecycleOwner)
    }

    val surfaceRequest by setupViewModel.pipeline.surfaceRequests.collectAsStateWithLifecycle()
    val bindState by setupViewModel.pipeline.bindState.collectAsStateWithLifecycle()
    val zoomRatio by setupViewModel.zoomRatio.collectAsStateWithLifecycle()

    val failed = bindState as? CameraPipeline.BindState.Failed
    if (failed != null) {
        CameraBindFailed(
            modifier = modifier,
            message = stringResource(R.string.camera_bind_failed, failed.error.message ?: ""),
            onRetry = { retryToken++ }
        )
    } else {
        CameraFrameBox(
            surfaceRequest = surfaceRequest,
            modifier = modifier,
            maxBoxWidth = maxBoxWidth
        ) { mapper ->
            // Pinch-to-zoom layer. Overlays with their own gesture handling (DragHandles) sit
            // above it and forward pinches themselves, because only the topmost sibling
            // receives pointer events.
            Box(
                modifier = Modifier
                    .matchParentSize()
                    .pointerInput(setupViewModel) {
                        detectTransformGestures { _, _, zoom, _ ->
                            if (zoom != 1f) setupViewModel.zoomBy(zoom)
                        }
                    }
            )
            overlay(mapper)
            if (surfaceRequest == null) {
                CircularProgressIndicator(modifier = Modifier.align(Alignment.Center))
            }
            SetupZoomIndicator(
                zoomRatio = zoomRatio,
                onReset = { setupViewModel.setZoomRatio(1f) },
                modifier = Modifier.align(Alignment.TopEnd).padding(8.dp)
            )
        }
    }
}

@Composable
private fun CameraBindFailed(message: String, onRetry: () -> Unit, modifier: Modifier = Modifier) {
    Box(modifier = modifier.background(Color.Black), contentAlignment = Alignment.Center) {
        Column(
            modifier = Modifier.padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(text = message, color = Color.White, textAlign = TextAlign.Center)
            Spacer(modifier = Modifier.height(12.dp))
            Button(onClick = onRetry) { Text(stringResource(R.string.action_retry)) }
        }
    }
}

@Composable
private fun SetupZoomIndicator(zoomRatio: Float, onReset: () -> Unit, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier
            .clip(RoundedCornerShape(8.dp))
            .background(Color.Black.copy(alpha = 0.55f))
            .padding(horizontal = 10.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = stringResource(R.string.camera_zoom, zoomRatio),
            color = Color.White,
            style = MaterialTheme.typography.labelMedium
        )
        if (zoomRatio != 1f) {
            Spacer(modifier = Modifier.width(8.dp))
            TextButton(
                onClick = onReset,
                contentPadding = PaddingValues(0.dp),
                colors = ButtonDefaults.textButtonColors(contentColor = Color.White)
            ) {
                Text(stringResource(R.string.setup_zoom_reset))
            }
        }
    }
}

/**
 * Shared "two draggable points" behaviour used by the ruler and region-of-interest screens: a
 * transparent gesture layer that adds a point on tap (while fewer than two are set), moves the
 * closest point on drag and forwards pinch gestures as zoom, plus the point handles themselves.
 */
@Composable
internal fun BoxScope.DragHandles(
    points: List<Point2D>,
    mapper: FrameMapper,
    onAddPoint: (Point2D) -> Unit,
    onMovePoint: (Int, Point2D) -> Unit,
    onZoom: (Float) -> Unit
) {
    val handleSlopPx = with(LocalDensity.current) { HandleTouchSlop.toPx() }
    var draggedIndex by remember { mutableStateOf(-1) }
    // Read through state holders so that the gesture coroutine is not restarted (and the
    // current drag cancelled) every time a point moves.
    val currentPoints by rememberUpdatedState(points)
    val currentMapper by rememberUpdatedState(mapper)
    val currentOnAddPoint by rememberUpdatedState(onAddPoint)
    val currentOnMovePoint by rememberUpdatedState(onMovePoint)
    val currentOnZoom by rememberUpdatedState(onZoom)

    Box(
        modifier = Modifier
            .matchParentSize()
            .pointerInput(Unit) {
                detectHandleGestures(
                    points = { currentPoints },
                    mapper = { currentMapper },
                    handleSlopPx = handleSlopPx,
                    onTap = { offset ->
                        if (currentPoints.size < 2) currentOnAddPoint(currentMapper.toCanonical(offset))
                    },
                    onDragStart = { index -> draggedIndex = index },
                    onDrag = { index, position -> currentOnMovePoint(index, currentMapper.toCanonical(position)) },
                    onDragEnd = { draggedIndex = -1 },
                    onZoom = { zoom -> currentOnZoom(zoom) }
                )
            }
    )
    points.forEachIndexed { index, point ->
        SetupHandle(center = mapper.toBox(point), isDragged = index == draggedIndex)
    }
}

private enum class HandleGestureMode { PENDING, DRAG, ZOOM, IGNORED }

/**
 * One detector for the three gestures of the handle layer, so that they cannot swallow each
 * other's events: a tap (no movement beyond the touch slop) adds a point, a single-finger drag
 * that starts on a handle moves it, and any two-finger gesture is treated as a pinch zoom.
 */
private suspend fun PointerInputScope.detectHandleGestures(
    points: () -> List<Point2D>,
    mapper: () -> FrameMapper,
    handleSlopPx: Float,
    onTap: (Offset) -> Unit,
    onDragStart: (Int) -> Unit,
    onDrag: (Int, Offset) -> Unit,
    onDragEnd: () -> Unit,
    onZoom: (Float) -> Unit
) {
    awaitEachGesture {
        val down = awaitFirstDown(requireUnconsumed = false)
        val handleIndex = closestPointIndex(points(), mapper(), down.position, handleSlopPx)
        var mode = HandleGestureMode.PENDING
        var travelled = Offset.Zero
        var dragStarted = false
        try {
            while (true) {
                val event = awaitPointerEvent()
                val pressed = event.changes.filter { it.pressed }
                if (pressed.isEmpty()) break

                if (pressed.size >= 2) {
                    mode = HandleGestureMode.ZOOM
                    val zoom = event.calculateZoom()
                    if (zoom != 1f) onZoom(zoom)
                    event.changes.forEach { if (it.positionChanged()) it.consume() }
                    continue
                }
                if (mode == HandleGestureMode.ZOOM || mode == HandleGestureMode.IGNORED) continue

                val change = event.changes.firstOrNull { it.id == down.id && it.pressed } ?: continue
                travelled += change.positionChange()
                if (mode == HandleGestureMode.PENDING && travelled.getDistance() > viewConfiguration.touchSlop) {
                    if (handleIndex >= 0) {
                        mode = HandleGestureMode.DRAG
                        dragStarted = true
                        onDragStart(handleIndex)
                    } else {
                        mode = HandleGestureMode.IGNORED
                    }
                }
                if (mode == HandleGestureMode.DRAG) {
                    onDrag(handleIndex, change.position)
                    change.consume()
                }
            }
            if (mode == HandleGestureMode.PENDING && handleIndex < 0) onTap(down.position)
        } finally {
            if (dragStarted) onDragEnd()
        }
    }
}

private fun closestPointIndex(points: List<Point2D>, mapper: FrameMapper, start: Offset, thresholdPx: Float): Int {
    var bestIndex = -1
    var bestDistance = Float.MAX_VALUE
    points.forEachIndexed { index, point ->
        val distance = (mapper.toBox(point) - start).getDistance()
        if (distance < bestDistance) {
            bestDistance = distance
            bestIndex = index
        }
    }
    return if (bestIndex >= 0 && bestDistance <= thresholdPx) bestIndex else -1
}

@Composable
private fun SetupHandle(center: Offset, isDragged: Boolean = false) {
    val handleSize = if (isDragged) HandleSizeDragged else HandleSize
    val borderWidth = if (isDragged) 3.dp else 2.dp
    val halfSizePx = with(LocalDensity.current) { handleSize.toPx() / 2f }
    Box(
        modifier = Modifier
            .offset { IntOffset((center.x - halfSizePx).roundToInt(), (center.y - halfSizePx).roundToInt()) }
            .size(handleSize)
            .clip(CircleShape)
            .background(Color.Transparent)
    ) {
        // The circle stays fully transparent so the image is visible under the crosshair. The white
        // crosshair and border each get a subtle dark shadow drawn just underneath/outside them so
        // they stay visible on bright backgrounds too.
        Canvas(modifier = Modifier.matchParentSize()) {
            val strokeWidthPx = HandleCrosshairWidth.toPx()
            val shadowStrokeWidthPx = HandleCrosshairShadowWidth.toPx()
            val borderWidthPx = borderWidth.toPx()
            val shadowRingWidthPx = HandleBorderShadowWidth.toPx()
            val centerOffset = Offset(size.width / 2f, size.height / 2f)

            drawLine(
                color = HandleShadowColor,
                start = Offset(centerOffset.x, 0f),
                end = Offset(centerOffset.x, size.height),
                strokeWidth = shadowStrokeWidthPx
            )
            drawLine(
                color = HandleShadowColor,
                start = Offset(0f, centerOffset.y),
                end = Offset(size.width, centerOffset.y),
                strokeWidth = shadowStrokeWidthPx
            )
            drawLine(
                color = Color.White,
                start = Offset(centerOffset.x, 0f),
                end = Offset(centerOffset.x, size.height),
                strokeWidth = strokeWidthPx
            )
            drawLine(
                color = Color.White,
                start = Offset(0f, centerOffset.y),
                end = Offset(size.width, centerOffset.y),
                strokeWidth = strokeWidthPx
            )

            val borderRadius = size.minDimension / 2f - borderWidthPx / 2f
            drawCircle(
                color = HandleShadowColor,
                radius = borderRadius + borderWidthPx / 2f + shadowRingWidthPx / 2f,
                center = centerOffset,
                style = Stroke(width = shadowRingWidthPx)
            )
            drawCircle(
                color = Color.White,
                radius = borderRadius,
                center = centerOffset,
                style = Stroke(width = borderWidthPx)
            )
        }
    }
}

/** Draws an axis-aligned rectangle outline (region of interest, tracked object boxes). */
@Composable
internal fun BoxScope.RectOutline(rect: Rect, color: Color, strokeWidth: Dp = 3.dp) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        drawRect(
            color = color,
            topLeft = rect.topLeft,
            size = rect.size,
            style = Stroke(width = strokeWidth.toPx())
        )
    }
}

/** Draws a line between two box-pixel offsets (e.g. the ruler reference line). */
@Composable
internal fun BoxScope.LineOverlay(from: Offset, to: Offset, color: Color, strokeWidth: Dp = 4.dp) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        drawLine(
            color = color,
            start = from,
            end = to,
            strokeWidth = strokeWidth.toPx(),
            cap = StrokeCap.Round
        )
    }
}
