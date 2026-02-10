package de.tudarmstadt.physics.trackingplot.ui.tracking_area

import android.graphics.RectF
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import de.tudarmstadt.physics.trackingplot.ui.ruler.CameraViewModel
import de.tudarmstadt.physics.trackingplot.ui.ruler.CameraWithOverlay

@Composable
fun TrackingAreaSelectionScreen(
    viewModel: CameraViewModel = viewModel(),
    onDone: (RectF) -> Unit
) {
    var rect by remember { mutableStateOf(RectF(200f, 200f, 600f, 600f)) }

    CameraWithOverlay { _ ->

        Canvas(modifier = Modifier
            .fillMaxSize()
//            .then( // You can implement resize handles or corner drags
//                // For simplicity here we just show the rect
//            )
        ) {
            drawRect(
                color = Color(0x80FF5722),
                topLeft = Offset(rect.left, rect.top),
                size = Size(rect.width(), rect.height()),
                style = Stroke(width = 6.dp.toPx())
            )
            drawRect(
                color = Color(0x4DFF5722),
                topLeft = Offset(rect.left, rect.top),
                size = Size(rect.width(), rect.height())
            )
        }

        // Very basic version — you should implement proper rectangle resize & move
        // (e.g. 4 corner handles + center drag)

        Button(
            onClick = {
                viewModel.trackingRect = rect
                onDone(rect)
            },
            modifier = Modifier
//                .align(Alignment.BottomCenter)
                .padding(24.dp)
        ) {
            Text("Start Tracking")
        }
    }
}