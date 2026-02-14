#include <jni.h>


#include "TrackerManager.hpp"

static TrackerManager gTrackerManager;

cv::Scalar rgbToHsv(int r, int g, int b);

extern "C"
JNIEXPORT jint JNICALL
Java_de_tudarmstadt_physics_trackingplot_tracker2_NativeTracker_addTracker(
//Java_com_example_tracker_NativeTracker_addTracker(
        JNIEnv* env,
        jobject,
//        jint h, jint s, jint v,
        jint r, jint g, jint b,
        jint tolerance) {

    cv::Scalar hsv = rgbToHsv(r, g, b);
//    cv::Scalar hsv(h, s, v);
    return gTrackerManager.addTracker(hsv, tolerance);
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_de_tudarmstadt_physics_trackingplot_tracker2_NativeTracker_updateTrackers(
        JNIEnv* env,
        jobject,
        jlong matAddr,
        jint roiX,
        jint roiY,
        jint roiWidth,
        jint roiHeight) {

    cv::Mat& frame = *(cv::Mat*)matAddr;

//    auto results = gTrackerManager.updateAll(frame);
    std::vector<TrackerResult> results;

    if (roiWidth > 0 && roiHeight > 0) {
        cv::Rect roi(roiX, roiY, roiWidth, roiHeight);
        results = gTrackerManager.updateAll(frame, &roi);
    } else {
        results = gTrackerManager.updateAll(frame, nullptr);
    }

    jclass resultClass = env->FindClass("de/tudarmstadt/physics/trackingplot/tracker2/TrackerResult");
    jobjectArray array = env->NewObjectArray(
            results.size(),
            resultClass,
            nullptr
    );

    jmethodID constructor = env->GetMethodID(
            resultClass,
            "<init>",
            "(IIIII)V"
    );

    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];

        jobject obj = env->NewObject(
                resultClass,
                constructor,
                r.trackerId,
                r.boundingBox.x,
                r.boundingBox.y,
                r.boundingBox.width,
                r.boundingBox.height
        );

        env->SetObjectArrayElement(array, i, obj);
    }

    return array;
}

cv::Scalar rgbToHsv(int r, int g, int b) {

    cv::Mat bgr(1, 1, CV_8UC3, cv::Scalar(b, g, r));
    cv::Mat hsv;

    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    cv::Vec3b pixel = hsv.at<cv::Vec3b>(0, 0);

    return cv::Scalar(pixel[0], pixel[1], pixel[2]);
}







//#include <jni.h>
//#include "TrackerManager.hpp"
//
//static TrackerManager* manager = nullptr;
//
//extern "C"
//JNIEXPORT void JNICALL
//Java_de_tudarmstadt_physics_trackingplot_tracker_NativeTracker_nativeInit(
//        JNIEnv* env,
//        jobject,
//        jlong matAddr,
//        jobjectArray configs,
//        jdoubleArray lowerHSVArr,
//        jdoubleArray upperHSVArr) {
//
//    if (manager == nullptr)
//        manager = new TrackerManager();
//
//    cv::Mat& frame = *(cv::Mat*)matAddr;
//
//    int count = env->GetArrayLength(configs);
//    std::vector<TrackerConfig> nativeConfigs;
//
//    for (int i = 0; i < count; i++) {
//
//        jobject obj = env->GetObjectArrayElement(configs, i);
//        jclass cls = env->GetObjectClass(obj);
//
//        TrackerConfig cfg;
//
//        cfg.id = env->GetIntField(obj,
//                env->GetFieldID(cls, "id", "I"));
//
//        double x = env->GetDoubleField(obj,
//                env->GetFieldID(cls, "x", "D"));
//        double y = env->GetDoubleField(obj,
//                env->GetFieldID(cls, "y", "D"));
//        double w = env->GetDoubleField(obj,
//                env->GetFieldID(cls, "width", "D"));
//        double h = env->GetDoubleField(obj,
//                env->GetFieldID(cls, "height", "D"));
//
//        cfg.initialBox = cv::Rect2d(x, y, w, h);
//        cfg.hasConstraint = false;
//
//        nativeConfigs.push_back(cfg);
//    }
//
//    jdouble* lowerPtr = env->GetDoubleArrayElements(lowerHSVArr, nullptr);
//    jdouble* upperPtr = env->GetDoubleArrayElements(upperHSVArr, nullptr);
//
//    cv::Scalar lower(lowerPtr[0], lowerPtr[1], lowerPtr[2]);
//    cv::Scalar upper(upperPtr[0], upperPtr[1], upperPtr[2]);
//
//    env->ReleaseDoubleArrayElements(lowerHSVArr, lowerPtr, 0);
//    env->ReleaseDoubleArrayElements(upperHSVArr, upperPtr, 0);
//
//    manager->init(frame, nativeConfigs, lower, upper);
//}
//
//extern "C"
//JNIEXPORT jobjectArray JNICALL
//Java_de_tudarmstadt_physics_trackingplot_tracker_NativeTracker_nativeUpdate(
//        JNIEnv* env,
//        jobject,
//        jlong matAddr) {
//
//    cv::Mat& frame = *(cv::Mat*)matAddr;
//    auto results = manager->update(frame);
//
//    jclass cls = env->FindClass("de/tudarmstadt/physics/trackingplot/tracker/TrackerResult");
//    jobjectArray arr =
//            env->NewObjectArray(results.size(), cls, nullptr);
//
//    jmethodID ctor =
//            env->GetMethodID(cls, "<init>", "(IDDDDZ)V");
//
//    for (int i = 0; i < results.size(); i++) {
//        auto& r = results[i];
//
//        jobject obj = env->NewObject(
//                cls, ctor,
//                r.id,
//                r.box.x,
//                r.box.y,
//                r.box.width,
//                r.box.height,
//                r.valid);
//
//        env->SetObjectArrayElement(arr, i, obj);
//    }
//
//    return arr;
//}



#include <opencv2/opencv.hpp>

using namespace cv;

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_de_tudarmstadt_physics_trackingplot_tracking_NativeTracker_processFrame(
        JNIEnv *env,
        jobject,
        jlong rgbaAddr,
        jint width,
        jint height) {

    Mat &rgba = *(Mat *) rgbaAddr;

    //START RED
    Mat hsv, mask1, mask2, mask;
    cvtColor(rgba, hsv, COLOR_RGB2HSV);

    inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), mask1);
    inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);
    bitwise_or(mask1, mask2, mask);
    //END RED
    //START GREEN
//    cv::Mat hsv, mask;
//    cvtColor(rgba, hsv, COLOR_RGB2HSV);

//    cv::inRange(
//        hsv,
//        cv::Scalar(35, 100, 80),   // H, S, V (lower)
//        cv::Scalar(85, 255, 255),  // H, S, V (upper)
//        mask
//    );
    //END GREEN
    // Noise reduction (VERY important)
    cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 1);
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);

    std::vector<std::vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty())
        return nullptr;

    // Find largest contour
    double maxArea = 0;
    int maxIdx = -1;

    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxIdx = i;
        }
    }

    if (maxIdx < 0)
        return nullptr;

    Moments m = moments(contours[maxIdx]);
    if (m.m00 == 0)
        return nullptr;

    float cx = static_cast<float>(m.m10 / m.m00);
    float cy = static_cast<float>(m.m01 / m.m00);

    // Normalize
    float nx = cx / width;
    float ny = cy / height;

    // Optional debug draw
    circle(rgba, Point(cx, cy), 10, Scalar(0, 255, 0), 2);

    jfloatArray result = env->NewFloatArray(2);
    jfloat tmp[2] = {nx, ny};
    env->SetFloatArrayRegion(result, 0, 2, tmp);

    return result;
}



extern "C"
JNIEXPORT jintArray JNICALL
Java_de_tudarmstadt_physics_trackingplot_tracking_NativeTracker_detectEdges(
        JNIEnv* env,
        jobject,
        jobject yBuf,
        jobject uBuf,
        jobject vBuf,
        jint width,
        jint height,
        jint yStride,
        jint uvStride,
        jint uvPixelStride
) {
    uint8_t* y = (uint8_t*)env->GetDirectBufferAddress(yBuf);
    uint8_t* u = (uint8_t*)env->GetDirectBufferAddress(uBuf);
    uint8_t* v = (uint8_t*)env->GetDirectBufferAddress(vBuf);

    cv::Mat yMat(height, width, CV_8UC1, y, yStride);
    cv::Mat uvMat(height / 2, width / 2, CV_8UC2);

    for (int i = 0; i < height / 2; i++) {
        for (int j = 0; j < width / 2; j++) {
            uvMat.at<cv::Vec2b>(i, j)[0] = u[i * uvStride + j * uvPixelStride];
            uvMat.at<cv::Vec2b>(i, j)[1] = v[i * uvStride + j * uvPixelStride];
        }
    }

    cv::Mat rgb;
    cv::cvtColorTwoPlane(yMat, uvMat, rgb, cv::COLOR_YUV2RGB);

    // ---- OpenCV processing ----
    cv::Mat gray, edges;
    cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
    cv::Canny(gray, edges, 80, 160);

    cv::Rect bbox = cv::boundingRect(edges);

    jintArray out = env->NewIntArray(4);
    jint data[4] = { bbox.x, bbox.y, bbox.width, bbox.height };
    env->SetIntArrayRegion(out, 0, 4, data);

    return out;
}




////#include <string>
//#include <android/log.h>
//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//
//#define TAG "NativeLib"
//
//using namespace std;
//using namespace cv;
//
//extern "C" {
//
///**
// * Native function called from Java/Kotlin to process camera frames
// * This applies OpenCV adaptive threshold to convert grayscale image to binary
// * @param env JNI environment
// * @param instance calling object instance
// * @param matAddr memory address of OpenCV Mat object from Java
// */
//void JNICALL
//Java_de_tudarmstadt_physics_trackingplot_MainActivity_adaptiveThresholdFromJNI(JNIEnv *env,
//        jobject instance,
//        jlong matAddr) {
//
//    // Get Mat from memory address passed from Java/Kotlin
//    Mat &mat = *(Mat *) matAddr;
//
//    // Record start time for performance measurement
//    clock_t begin = clock();
//
//    // Apply OpenCV adaptive threshold
//    // Parameters: input/output mat, max value, adaptive method, threshold type, block size, constant
//    cv::adaptiveThreshold(mat, mat, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 10);
//
//    // Calculate and log processing time
//    double totalTime = double(clock() - begin) / CLOCKS_PER_SEC;
//    __android_log_print(ANDROID_LOG_INFO, TAG, "adaptiveThreshold computation time = %f seconds\n",
//            totalTime);
//}

//void JNICALL
//Java_de_tudarmstadt_physics_trackingplot_MainActivity_highlightRedDot(
//    JNIEnv *env,
//    jobject instance,
//    jlong matAddr
//) {
//    // Get Mat from memory address passed from Java/Kotlin
//    Mat &frame = *(Mat *) matAddr;
//
//    // Define the tracking area (ROI) - e.g., center half of the frame
//    int width = frame.cols;
//    int height = frame.rows;
//    Rect roi(width / 4, height / 4, width / 2, height / 2);
//    // Draw the ROI rectangle (red, thickness 2)
//    rectangle(frame, roi, Scalar(255, 0, 0, 255), 2);           // Red ROI border
//
//    Mat roiMat = frame(roi);
//
//    Mat hsv;
//    cvtColor(roiMat, hsv, COLOR_RGB2HSV);
//
//
///* BLACK DOT TRACKING
//    // Threshold for BLACK color
//    // Black = very low Value (brightness), Hue & Saturation can be almost anything
//    Mat mask;
//    inRange(hsv,
//            Scalar(0,   0,   0),     // lower bound
//            Scalar(180, 255, 40),    // upper bound - adjust 40-60 depending on lighting
//            mask);
//    // Clean up the mask (very important for black detection!)
//    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
//    morphologyEx(mask, mask, MORPH_OPEN, kernel, Point(-1,-1), 2);   // remove small noise
//    morphologyEx(mask, mask, MORPH_CLOSE, kernel, Point(-1,-1), 1);  // fill small holes
//*/
//
//// Red color ranges in HSV (two ranges because red wraps around)
//    Mat mask1, mask2, mask;
////    inRange(hsv, Scalar(0,   120, 70),  Scalar(10,  255, 255), mask1);   // Lower red
////    inRange(hsv, Scalar(165, 120, 70),  Scalar(180, 255, 255), mask2);   // Upper red
////    bitwise_or(mask1, mask2, mask);
//    inRange(hsv, Scalar(35, 120, 70), Scalar(85, 255, 255), mask);
//
//    // Clean up the mask - very important for stable detection
//    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
//    morphologyEx(mask, mask, MORPH_OPEN,  kernel, Point(-1,-1), 1);  // remove small noise
//    morphologyEx(mask, mask, MORPH_CLOSE, kernel, Point(-1,-1), 1);  // fill small holes
//
//
//
//
//    // Find contours
//    vector<vector<Point>> contours;
//    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//
//    // Assume the largest contour is the red dot (single dot)
//    if (!contours.empty()) {
//        // Find the largest contour by area
//        double maxArea = 0;
//        int maxIdx = -1;
//        for (int i = 0; i < contours.size(); i++) {
//            double area = contourArea(contours[i]);
//            if (area > maxArea) {
//                maxArea = area;
//                maxIdx = i;
//            }
//        }
//
//        if (maxIdx != -1) {
//            // Get bounding box (relative to ROI)
//            Rect bounding = boundingRect(contours[maxIdx]);
//
//            // Adjust bounding box to original frame coordinates
//            bounding.x += roi.x;
//            bounding.y += roi.y;
//
//            // Draw the bounding box on the original frame (green, thickness 2)
//            rectangle(frame, bounding, Scalar(255, 0, 0, 255), 2);
//        }
//    }
//}
//
//}