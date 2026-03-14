#include <jni.h>


#include "TrackerManager.hpp"

static TrackerManager gTrackerManager;

cv::Scalar rgbToHsv(int r, int g, int b);

extern "C"
JNIEXPORT jint JNICALL
Java_de_tudarmstadt_physics_trackingplot_tracker2_NativeTracker_addTracker(
        JNIEnv* env,
        jobject,
        jint r, jint g, jint b,
        jint tolerance) {

    cv::Scalar hsv = rgbToHsv(r, g, b);
    return gTrackerManager.addTracker(hsv, tolerance);
}

extern "C"
JNIEXPORT jint JNICALL
Java_de_tudarmstadt_physics_trackingplot_tracker2_NativeTracker_setTracker(
        JNIEnv* env,
        jobject,
        jint index,
        jint r, jint g, jint b,
        jint tolerance) {

    cv::Scalar hsv = rgbToHsv(r, g, b);
    return gTrackerManager.setTracker(index, hsv, tolerance);
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

    std::vector<TrackerResult> results;

    if (roiWidth > 0 && roiHeight > 0) {
        cv::Rect roi(roiX, roiY, roiWidth, roiHeight);
        results = gTrackerManager.updateAll(frame, &roi);
    } else {
        results = gTrackerManager.updateAll(frame, nullptr);
    }

    jclass resultClass = env->FindClass("de/tudarmstadt/physics/trackingplot/tracker/TrackerResult");
    jobjectArray array = env->NewObjectArray(
            results.size(),
            resultClass,
            nullptr
    );

    jmethodID constructor = env->GetMethodID(
            resultClass,
            "<init>",
            "(IZIIIIDD)V"
    );

    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];

        jobject obj = env->NewObject(
                resultClass,
                constructor,
                r.trackerId,
                r.boundingBoxAndCentroid.has_value(),
                r.boundingBoxAndCentroid->first.x,
                r.boundingBoxAndCentroid->first.y,
                r.boundingBoxAndCentroid->first.width,
                r.boundingBoxAndCentroid->first.height,
                r.boundingBoxAndCentroid->second.x,
                r.boundingBoxAndCentroid->second.y
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


extern "C"
JNIEXPORT void JNICALL
Java_de_tudarmstadt_physics_trackingplot_tracker2_NativeTracker_reset(
        JNIEnv* env,
        jobject) {
    gTrackerManager.reset();
}