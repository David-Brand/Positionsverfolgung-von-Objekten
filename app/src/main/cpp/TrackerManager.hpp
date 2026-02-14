#ifndef TRACKINGPLOT_TRACKERMANAGER_HPP
#define TRACKINGPLOT_TRACKERMANAGER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

struct TrackerResult {
    int trackerId;
    cv::Rect boundingBox;
};

class ColorTracker {
public:
    ColorTracker(int id, cv::Scalar hsvColor, int tolerance);

    TrackerResult update(const cv::Mat& frame);

private:
    int trackerId;
    cv::Scalar targetColor;
    int tolerance;
};

class TrackerManager {
public:
    int addTracker(cv::Scalar hsvColor, int tolerance);
    std::vector<TrackerResult> updateAll(
            const cv::Mat& frame,
            const cv::Rect* roi = nullptr);

private:
    int nextId = 0;
    std::vector<ColorTracker> trackers;
};





////#pragma once
//
//#include <opencv2/opencv.hpp>
//#include <vector>
//
//struct TrackerConfig {
//    int id;
//    cv::Rect2d initialBox;
//    cv::Rect2d constraintBox;
//    bool hasConstraint;
//};
//
//struct TrackerResult {
//    int id;
//    cv::Rect2d box;
//    bool valid;
//};
//
//class TrackerManager {
//public:
//    TrackerManager();
//    ~TrackerManager();
//
//    void init(const cv::Mat& frame,
//            const std::vector<TrackerConfig>& configs,
//            cv::Scalar lowerHSV,
//            cv::Scalar upperHSV);
//
//    std::vector<TrackerResult> update(const cv::Mat& frame);
//    void clear();
//
//private:
//    struct InternalTracker {
//        int id;
//        cv::Rect2d box;
//        cv::Rect2d constraintBox;
//        bool hasConstraint;
//        bool valid;
//        int lostFrames;
//    };
//
//    std::vector<cv::Rect> detectColorBlobs(const cv::Mat& frame);
//    void associate(std::vector<cv::Rect>& detections);
//
//    std::vector<InternalTracker> trackers;
//
//    cv::Scalar lowerHSV;
//    cv::Scalar upperHSV;
//};


#endif //TRACKINGPLOT_TRACKERMANAGER_HPP
