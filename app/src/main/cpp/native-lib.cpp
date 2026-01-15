#include <jni.h>
//#include <string>
#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define TAG "NativeLib"

using namespace std;
using namespace cv;

extern "C" {

/**
 * Native function called from Java/Kotlin to process camera frames
 * This applies OpenCV adaptive threshold to convert grayscale image to binary
 * @param env JNI environment
 * @param instance calling object instance
 * @param matAddr memory address of OpenCV Mat object from Java
 */
void JNICALL
Java_de_tudarmstadt_physics_trackingplot_MainActivity_adaptiveThresholdFromJNI(JNIEnv *env,
        jobject instance,
        jlong matAddr) {

    // Get Mat from memory address passed from Java/Kotlin
    Mat &mat = *(Mat *) matAddr;

    // Record start time for performance measurement
    clock_t begin = clock();

    // Apply OpenCV adaptive threshold
    // Parameters: input/output mat, max value, adaptive method, threshold type, block size, constant
    cv::adaptiveThreshold(mat, mat, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 10);

    // Calculate and log processing time
    double totalTime = double(clock() - begin) / CLOCKS_PER_SEC;
    __android_log_print(ANDROID_LOG_INFO, TAG, "adaptiveThreshold computation time = %f seconds\n",
            totalTime);
}

void JNICALL
Java_de_tudarmstadt_physics_trackingplot_MainActivity_highlightRedDot(
    JNIEnv *env,
    jobject instance,
    jlong matAddr
) {
    // Get Mat from memory address passed from Java/Kotlin
    Mat &frame = *(Mat *) matAddr;

    // Define the tracking area (ROI) - e.g., center half of the frame
    int width = frame.cols;
    int height = frame.rows;
    Rect roi(width / 4, height / 4, width / 2, height / 2);
    // Draw the ROI rectangle (red, thickness 2)
    rectangle(frame, roi, Scalar(255, 0, 0, 255), 2);           // Red ROI border

    Mat roiMat = frame(roi);

    Mat hsv;
    cvtColor(roiMat, hsv, COLOR_RGB2HSV);


/* BLACK DOT TRACKING
    // Threshold for BLACK color
    // Black = very low Value (brightness), Hue & Saturation can be almost anything
    Mat mask;
    inRange(hsv,
            Scalar(0,   0,   0),     // lower bound
            Scalar(180, 255, 40),    // upper bound - adjust 40-60 depending on lighting
            mask);
    // Clean up the mask (very important for black detection!)
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel, Point(-1,-1), 2);   // remove small noise
    morphologyEx(mask, mask, MORPH_CLOSE, kernel, Point(-1,-1), 1);  // fill small holes
*/

// Red color ranges in HSV (two ranges because red wraps around)
    Mat mask1, mask2, mask;
    inRange(hsv, Scalar(0,   120, 70),  Scalar(10,  255, 255), mask1);   // Lower red
    inRange(hsv, Scalar(165, 120, 70),  Scalar(180, 255, 255), mask2);   // Upper red
    bitwise_or(mask1, mask2, mask);

    // Clean up the mask - very important for stable detection
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN,  kernel, Point(-1,-1), 1);  // remove small noise
    morphologyEx(mask, mask, MORPH_CLOSE, kernel, Point(-1,-1), 1);  // fill small holes




    // Find contours
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Assume the largest contour is the red dot (single dot)
    if (!contours.empty()) {
        // Find the largest contour by area
        double maxArea = 0;
        int maxIdx = -1;
        for (int i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
                maxIdx = i;
            }
        }

        if (maxIdx != -1) {
            // Get bounding box (relative to ROI)
            Rect bounding = boundingRect(contours[maxIdx]);

            // Adjust bounding box to original frame coordinates
            bounding.x += roi.x;
            bounding.y += roi.y;

            // Draw the bounding box on the original frame (green, thickness 2)
            rectangle(frame, bounding, Scalar(0, 255, 0, 255), 2);
        }
    }
}

}