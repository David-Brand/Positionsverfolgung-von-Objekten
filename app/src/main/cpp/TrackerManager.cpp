#include "TrackerManager.hpp"

ColorTracker::ColorTracker(int id, cv::Scalar hsvColor, int tol)
        : trackerId(id), targetColor(hsvColor), tolerance(tol) {}

TrackerResult ColorTracker::update(const cv::Mat& frame) {

    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_RGB2HSV);

    cv::Scalar lower(
            targetColor[0] - tolerance,
            targetColor[1] - tolerance,
            targetColor[2] - tolerance
    );

    cv::Scalar upper(
            targetColor[0] + tolerance,
            targetColor[1] + tolerance,
            targetColor[2] + tolerance
    );

    cv::Mat mask;
    cv::inRange(hsv, lower, upper, mask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::pair<cv::Rect, cv::Point2d> bestBoundingBoxAndCentroid;
    double maxArea = 0;

    for (auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;

            cv::Moments M = cv::moments(contour);

            if (M.m00 != 0) {
                double cx = M.m10 / M.m00;
                double cy = M.m01 / M.m00;

                bestBoundingBoxAndCentroid = std::make_pair(cv::boundingRect(contour), cv::Point2d(cx, cy));
            }
        }
    }

    if (maxArea > 0)
        return { trackerId, bestBoundingBoxAndCentroid };
    else
        return { trackerId, std::nullopt };
}

int TrackerManager::addTracker(cv::Scalar hsvColor, int tolerance) {
    int id = nextId++;
    trackers.emplace_back(id, hsvColor, tolerance);
    return id;
}

int TrackerManager::setTracker(int index, cv::Scalar hsvColor, int tolerance) {
//    int id = index;//nextId++;
//    trackers.emplace(trackers.begin() + index, hsvColor, tolerance);
//    trackers.emplace_back(id, hsvColor, tolerance);

    auto it = std::find_if(trackers.begin(), trackers.end(),
            [&](const ColorTracker& colorTracker) {
        return colorTracker.trackerId == index;
    });
    if (it != trackers.end()) {
        *it = ColorTracker(index, hsvColor, tolerance);
    } else {
        trackers.push_back(ColorTracker(index, hsvColor, tolerance));
    }
    return index;
}

std::vector<TrackerResult> TrackerManager::updateAll(
        const cv::Mat& frame,
        const cv::Rect* roi) {

    std::vector<TrackerResult> results;

    cv::Rect safeRoi;

    if (roi != nullptr && roi->width > 0 && roi->height > 0) {
        safeRoi = (*roi) & cv::Rect(0, 0, frame.cols, frame.rows);
    } else {
        safeRoi = cv::Rect(0, 0, frame.cols, frame.rows);
    }

    if (safeRoi.width <= 0 || safeRoi.height <= 0)
        return results;

    cv::Mat cropped = frame(safeRoi);

    for (auto& tracker : trackers) {

        TrackerResult r = tracker.update(cropped);

        if (r.boundingBoxAndCentroid) {
            // Offset only if ROI was used
            r.boundingBoxAndCentroid->first.x += safeRoi.x;
            r.boundingBoxAndCentroid->first.y += safeRoi.y;
            r.boundingBoxAndCentroid->second.x += safeRoi.x;
            r.boundingBoxAndCentroid->second.y += safeRoi.y;
        }

        results.push_back(r);
    }

    return results;
}

void TrackerManager::reset() {
    trackers.clear();
}