#ifndef TRACKINGPLOT_TRACKERMANAGER_HPP
#define TRACKINGPLOT_TRACKERMANAGER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

struct TrackerResult {
    int trackerId;
    std::optional<std::pair<cv::Rect, cv::Point2d>> boundingBoxAndCentroid;
};

class ColorTracker {
public:
    ColorTracker(int id, cv::Scalar hsvColor, int tolerance);

    TrackerResult update(const cv::Mat& frame);

    int trackerId;
private:
    cv::Scalar targetColor;
    int tolerance;
};

class TrackerManager {
public:
    int addTracker(cv::Scalar hsvColor, int tolerance);
    int setTracker(int index, cv::Scalar hsvColor, int tolerance);
    std::vector<TrackerResult> updateAll(
            const cv::Mat& frame,
            const cv::Rect* roi = nullptr);
    void reset();

private:
    int nextId = 0;
    std::vector<ColorTracker> trackers;
};

#endif //TRACKINGPLOT_TRACKERMANAGER_HPP
