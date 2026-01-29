#include <jni.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <cmath>
#include <limits>

using namespace cv;

struct HueRange {
    int center = 0;   // 0..179
    int tol = 12;     // +/- tolerance
    int minS = 80;    // saturation threshold
    int minV = 60;    // value threshold
};

struct SimpleState {
    std::vector<Point2f> prevCenters; // ROI-local
    std::vector<HueRange> hues;       // per object
    bool initialized = false;
};

static jfieldID getNativePtrField(JNIEnv* env, jobject thiz) {
    jclass cls = env->GetObjectClass(thiz);
    return env->GetFieldID(cls, "nativeTrackerPtr", "J");
}
static SimpleState* getState(JNIEnv* env, jobject thiz) {
    jlong p = env->GetLongField(thiz, getNativePtrField(env, thiz));
    return reinterpret_cast<SimpleState*>(p);
}
static void setState(JNIEnv* env, jobject thiz, SimpleState* st) {
    env->SetLongField(thiz, getNativePtrField(env, thiz), reinterpret_cast<jlong>(st));
}
static void clearState(JNIEnv* env, jobject thiz) {
    SimpleState* st = getState(env, thiz);
    if (st) { delete st; setState(env, thiz, nullptr); }
}

// Circular mean for hue values in [0..179]
static int circularMeanHue(const std::vector<int>& hues) {
    if (hues.empty()) return 0;
    double sumSin = 0.0, sumCos = 0.0;
    for (int h : hues) {
        double ang = (2.0 * CV_PI * h) / 180.0;
        sumSin += std::sin(ang);
        sumCos += std::cos(ang);
    }
    double meanAng = std::atan2(sumSin, sumCos);
    if (meanAng < 0) meanAng += 2.0 * CV_PI;
    int meanHue = (int)std::lround((meanAng * 180.0) / (2.0 * CV_PI));
    if (meanHue >= 180) meanHue -= 180;
    return meanHue;
}

// Circular “distance” on hue circle [0..179]
static int hueDist(int a, int b) {
    int d = std::abs(a - b);
    return std::min(d, 180 - d);
}

// Sample a patch around ROI-local point and derive hue center + tolerance
static bool initHueFromPatchRGBA(
        const Mat& roiRGBA,
        Point2f roiLocalCenter,
        HueRange& outHue,
        int patchRadiusPx = 4 // patch size = (2r+1)^2
) {
    if (roiRGBA.empty()) return false;

    int cx = (int)std::lround(roiLocalCenter.x);
    int cy = (int)std::lround(roiLocalCenter.y);

    int x0 = std::max(0, cx - patchRadiusPx);
    int y0 = std::max(0, cy - patchRadiusPx);
    int x1 = std::min(roiRGBA.cols - 1, cx + patchRadiusPx);
    int y1 = std::min(roiRGBA.rows - 1, cy + patchRadiusPx);

    Rect patch(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    if (patch.width <= 0 || patch.height <= 0) return false;

    Mat hsv;
    cvtColor(roiRGBA(patch), hsv, COLOR_RGB2HSV);

    std::vector<int> hueSamples;
    hueSamples.reserve((size_t)patch.area());

    // take pixels with decent saturation/value (avoid background)
    for (int y = 0; y < hsv.rows; y++) {
        const Vec3b* row = hsv.ptr<Vec3b>(y);
        for (int x = 0; x < hsv.cols; x++) {
            int H = row[x][0];
            int S = row[x][1];
            int V = row[x][2];
            if (S >= 60 && V >= 40) hueSamples.push_back(H);
        }
    }

    if (hueSamples.size() < 10) return false;

    int center = circularMeanHue(hueSamples);

    // estimate tolerance from sample spread
    int sumD = 0;
    int maxD = 0;
    for (int h : hueSamples) {
        int d = hueDist(h, center);
        sumD += d;
        maxD = std::max(maxD, d);
    }
    double meanD = (double)sumD / (double)hueSamples.size();

    // heuristic tolerance: based on mean spread, with caps
    int tol = (int)std::lround(std::max(8.0, std::min(30.0, meanD * 2.5 + 6.0)));
    // if the object is very uniform, tol stays small; if lighting varies, tol grows.

    outHue.center = center;
    outHue.tol = tol;
    outHue.minS = 80;
    outHue.minV = 60;
    return true;
}

// Create a mask for a single object's hue range (wrap-around aware)
static void maskForHueRange(const Mat& hsv, const HueRange& hr, Mat& outMask) {
    int low = hr.center - hr.tol;
    int high = hr.center + hr.tol;

    Mat m1, m2;
    if (low < 0) {
        // [0..high] OR [low+180 .. 179]
        inRange(hsv, Scalar(0, hr.minS, hr.minV), Scalar(high, 255, 255), m1);
        inRange(hsv, Scalar(low + 180, hr.minS, hr.minV), Scalar(179, 255, 255), m2);
        bitwise_or(m1, m2, outMask);
    } else if (high > 179) {
        // [low..179] OR [0..high-180]
        inRange(hsv, Scalar(low, hr.minS, hr.minV), Scalar(179, 255, 255), m1);
        inRange(hsv, Scalar(0, hr.minS, hr.minV), Scalar(high - 180, 255, 255), m2);
        bitwise_or(m1, m2, outMask);
    } else {
        inRange(hsv, Scalar(low, hr.minS, hr.minV), Scalar(high, 255, 255), outMask);
    }

    // Clean up noise
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
    morphologyEx(outMask, outMask, MORPH_OPEN,  kernel, Point(-1,-1), 1);
    morphologyEx(outMask, outMask, MORPH_CLOSE, kernel, Point(-1,-1), 1);
}

// Find best blob nearest previous center for one object
static bool findNearestBlob(
        const Mat& mask,
        const Point2f& prevCenter,
        Point2f& outCenter,
        float& outRadius,
        float maxDistPx
) {
    std::vector<std::vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    float bestD2 = maxDistPx * maxDistPx;
    bool found = false;

    for (auto& c : contours) {
        double area = contourArea(c);
        if (area < 8.0) continue;

        Point2f center;
        float radius;
        minEnclosingCircle(c, center, radius);
        if (radius < 1.5f) continue;

        Point2f d = center - prevCenter;
        float d2 = d.dot(d);
        if (d2 < bestD2) {
            bestD2 = d2;
            outCenter = center;
            outRadius = radius;
            found = true;
        }
    }
    return found;
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_de_tudarmstadt_physics_trackingplot_MainActivity_nativeTrack(
        JNIEnv* env,
        jobject thiz,
        jlong matAddr,
        jintArray roiArray,
        jintArray boxesInOutArray,
        jboolean reinit,
        jint hueInitIndex,
        jintArray hueInitPointArray // [x,y] full-frame coords
) {
    if (!matAddr || !roiArray || !boxesInOutArray) return JNI_FALSE;

    Mat& frame = *(Mat*)matAddr;
    if (frame.empty()) return JNI_FALSE;

    // ROI
    if (env->GetArrayLength(roiArray) < 4) return JNI_FALSE;
    jint rv[4];
    env->GetIntArrayRegion(roiArray, 0, 4, rv);
    Rect roi(rv[0], rv[1], rv[2], rv[3]);
    roi &= Rect(0, 0, frame.cols, frame.rows);
    if (roi.width <= 0 || roi.height <= 0) return JNI_FALSE;

    // Boxes
    jsize len = env->GetArrayLength(boxesInOutArray);
    if (len <= 0 || (len % 4) != 0) return JNI_FALSE;
    int n = (int)(len / 4);

    std::vector<jint> boxBuf((size_t)len);
    env->GetIntArrayRegion(boxesInOutArray, 0, len, boxBuf.data());

    // State
    SimpleState* st = getState(env, thiz);
    if (!st) { st = new SimpleState(); setState(env, thiz, st); }

    // Ensure vectors sized
    if ((int)st->prevCenters.size() != n) {
        st->prevCenters.assign(n, Point2f());
        st->hues.assign(n, HueRange());
        st->initialized = false;
    }

    Mat roiMat = frame(roi);

    // Update centers from current boxes if reinit or uninitialized
    if (reinit || !st->initialized) {
        for (int i = 0; i < n; i++) {
            float x = (float)boxBuf[i*4 + 0];
            float y = (float)boxBuf[i*4 + 1];
            float w = (float)boxBuf[i*4 + 2];
            float h = (float)boxBuf[i*4 + 3];
            st->prevCenters[i] = Point2f((x + 0.5f*w) - roi.x, (y + 0.5f*h) - roi.y);
        }
        st->initialized = true;
    }

    // If caller requested hue init for a specific object (user just confirmed)
    if (hueInitIndex >= 0 && hueInitIndex < n && hueInitPointArray && env->GetArrayLength(hueInitPointArray) >= 2) {
        jint p[2];
        env->GetIntArrayRegion(hueInitPointArray, 0, 2, p);

        // Convert to ROI-local
        Point2f roiLocal((float)(p[0] - roi.x), (float)(p[1] - roi.y));
        // Keep center consistent with selection point
        st->prevCenters[hueInitIndex] = roiLocal;

        HueRange hr;
        if (initHueFromPatchRGBA(roiMat, roiLocal, hr)) {
            st->hues[hueInitIndex] = hr;
        }
        // If sampling fails, we keep previous/default hue range.
    }

    // Prepare HSV once per frame
    Mat hsv;
    cvtColor(roiMat, hsv, COLOR_RGB2HSV);

    const float maxDistPx = 120.0f; // tune for your object speed
    bool updatedAny = false;

    // Track each object using its own hue range
    for (int i = 0; i < n; i++) {
        Mat mask;
        maskForHueRange(hsv, st->hues[i], mask);

        Point2f newC;
        float newR = 0.f;
        if (!findNearestBlob(mask, st->prevCenters[i], newC, newR, maxDistPx)) {
            continue; // keep last box if not found
        }

        st->prevCenters[i] = newC;
        updatedAny = true;

        float pad = 2.0f;
        float x = newC.x - newR - pad;
        float y = newC.y - newR - pad;
        float w = 2.0f * (newR + pad);
        float h = 2.0f * (newR + pad);

        // Clamp to ROI
        x = std::max(0.f, x);
        y = std::max(0.f, y);
        w = std::min(w, (float)roi.width - x);
        h = std::min(h, (float)roi.height - y);

        // Write back full-frame coords
        boxBuf[i*4 + 0] = (jint)lroundf(x + roi.x);
        boxBuf[i*4 + 1] = (jint)lroundf(y + roi.y);
        boxBuf[i*4 + 2] = (jint)lroundf(w);
        boxBuf[i*4 + 3] = (jint)lroundf(h);
    }

    // Draw ROI + boxes (red)
    rectangle(frame, roi, Scalar(255, 0, 0, 255), 2);
    for (int i = 0; i < n; i++) {
        Rect b(boxBuf[i*4 + 0], boxBuf[i*4 + 1], boxBuf[i*4 + 2], boxBuf[i*4 + 3]);
        b &= Rect(0, 0, frame.cols, frame.rows);
        if (b.width > 0 && b.height > 0) rectangle(frame, b, Scalar(255, 0, 0, 255), 2);
    }

    env->SetIntArrayRegion(boxesInOutArray, 0, len, boxBuf.data());
    return updatedAny ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_de_tudarmstadt_physics_trackingplot_MainActivity_nativeRelease(JNIEnv* env, jobject thiz) {
    clearState(env, thiz);
}

} // extern "C"
