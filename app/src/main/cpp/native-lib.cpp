#include <android/log.h>
#include <jni.h>

#include <opencv2/core.hpp>

#include <exception>
#include <vector>

#include "TrackerManager.hpp"

// JNI bridge for de.tudarmstadt.physics.trackingplot.tracker.NativeTracker.
//
// Each NativeTracker instance owns one TrackerManager (identified by an opaque handle), so the
// setup screen and the live experiment never share tracking state. Results are returned as a
// flat double array (8 values per tracker: id, found, x, y, width, height, centroidX,
// centroidY) to avoid per-frame reflection on the Java side.

namespace {

constexpr const char* kLogTag = "native-lib";
constexpr jsize kValuesPerResult = 8;

TrackerManager* manager(jlong handle) {
    return reinterpret_cast<TrackerManager*>(handle);
}

jdoubleArray emptyArray(JNIEnv* env) {
    return env->NewDoubleArray(0);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_de_tudarmstadt_physics_trackingplot_tracker_NativeTracker_nativeCreate(JNIEnv*, jclass) {
    return reinterpret_cast<jlong>(new TrackerManager());
}

extern "C" JNIEXPORT void JNICALL
Java_de_tudarmstadt_physics_trackingplot_tracker_NativeTracker_nativeDestroy(JNIEnv*, jclass, jlong handle) {
    delete manager(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_de_tudarmstadt_physics_trackingplot_tracker_NativeTracker_nativeSetTracker(
        JNIEnv*, jclass, jlong handle, jint id,
        jint h, jint s, jint v,
        jint hueTolerance, jint satTolerance, jint valTolerance) {
    if (handle == 0) return;
    manager(handle)->setTracker(id, HsvColor{h, s, v}, HsvTolerance{hueTolerance, satTolerance, valTolerance});
}

extern "C" JNIEXPORT jboolean JNICALL
Java_de_tudarmstadt_physics_trackingplot_tracker_NativeTracker_nativeRemoveTracker(
        JNIEnv*, jclass, jlong handle, jint id) {
    if (handle == 0) return JNI_FALSE;
    return manager(handle)->removeTracker(id) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_de_tudarmstadt_physics_trackingplot_tracker_NativeTracker_nativeClear(JNIEnv*, jclass, jlong handle) {
    if (handle == 0) return;
    manager(handle)->clear();
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_de_tudarmstadt_physics_trackingplot_tracker_NativeTracker_nativeUpdate(
        JNIEnv* env, jclass, jlong handle, jlong matAddr,
        jint roiX, jint roiY, jint roiWidth, jint roiHeight) {
    if (handle == 0 || matAddr == 0) {
        return emptyArray(env);
    }
    try {
        const cv::Mat& frame = *reinterpret_cast<cv::Mat*>(matAddr);
        const cv::Rect roi(roiX, roiY, roiWidth, roiHeight);
        const cv::Rect* roiPtr = (roiWidth > 0 && roiHeight > 0) ? &roi : nullptr;

        const std::vector<TrackerResult> results = manager(handle)->updateAll(frame, roiPtr);

        std::vector<jdouble> flat;
        flat.reserve(results.size() * kValuesPerResult);
        for (const TrackerResult& r : results) {
            flat.push_back(static_cast<jdouble>(r.trackerId));
            flat.push_back(r.found ? 1.0 : 0.0);
            flat.push_back(r.found ? static_cast<jdouble>(r.boundingBox.x) : 0.0);
            flat.push_back(r.found ? static_cast<jdouble>(r.boundingBox.y) : 0.0);
            flat.push_back(r.found ? static_cast<jdouble>(r.boundingBox.width) : 0.0);
            flat.push_back(r.found ? static_cast<jdouble>(r.boundingBox.height) : 0.0);
            flat.push_back(r.found ? r.centroid.x : 0.0);
            flat.push_back(r.found ? r.centroid.y : 0.0);
        }

        jdoubleArray array = env->NewDoubleArray(static_cast<jsize>(flat.size()));
        if (array == nullptr) {
            return nullptr;  // OutOfMemoryError is pending on the Java side
        }
        if (!flat.empty()) {
            env->SetDoubleArrayRegion(array, 0, static_cast<jsize>(flat.size()), flat.data());
        }
        return array;
    } catch (const cv::Exception& e) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag, "OpenCV error while tracking: %s", e.what());
    } catch (const std::exception& e) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag, "Error while tracking: %s", e.what());
    }
    return emptyArray(env);
}
