#include "TrackerManager.hpp"

ColorTracker::ColorTracker(int id, cv::Scalar hsvColor, int tol)
        : trackerId(id), targetColor(hsvColor), tolerance(tol) {}

TrackerResult ColorTracker::update(const cv::Mat& frame) {

    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

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

    cv::Rect bestRect;
    double maxArea = 0;

    for (auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            bestRect = cv::boundingRect(contour);
        }
    }

    return { trackerId, bestRect };
}

int TrackerManager::addTracker(cv::Scalar hsvColor, int tolerance) {
    int id = nextId++;
    trackers.emplace_back(id, hsvColor, tolerance);
    return id;
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

        // Offset only if ROI was used
        r.boundingBox.x += safeRoi.x;
        r.boundingBox.y += safeRoi.y;

        results.push_back(r);
    }

    return results;
//    std::vector<TrackerResult> results;
//
//    for (auto& tracker : trackers) {
//        results.push_back(tracker.update(frame));
//    }
//
//    return results;
}




//#include "TrackerManager.hpp"
//
//TrackerManager::TrackerManager() {}
//TrackerManager::~TrackerManager() { clear(); }
//
//void TrackerManager::clear() {
//    trackers.clear();
//}
//
//void TrackerManager::init(const cv::Mat& frame,
//        const std::vector<TrackerConfig>& configs,
//        cv::Scalar lower,
//        cv::Scalar upper) {
//
//    clear();
//
//    lowerHSV = lower;
//    upperHSV = upper;
//
//    for (const auto& cfg : configs) {
//        InternalTracker t;
//        t.id = cfg.id;
//        t.box = cfg.initialBox;
//        t.constraintBox = cfg.constraintBox;
//        t.hasConstraint = cfg.hasConstraint;
//        t.valid = true;
//        t.lostFrames = 0;
//        trackers.push_back(t);
//    }
//}
//
//std::vector<cv::Rect> TrackerManager::detectColorBlobs(
//        const cv::Mat& frame) {
//
//    cv::Mat hsv;
//    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
//
//    cv::Mat mask;
//    cv::inRange(hsv, lowerHSV, upperHSV, mask);
//
//    cv::erode(mask, mask, cv::Mat(), cv::Point(-1,-1), 2);
//    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1,-1), 2);
//
//    std::vector<std::vector<cv::Point>> contours;
//    cv::findContours(mask, contours,
//            cv::RETR_EXTERNAL,
//            cv::CHAIN_APPROX_SIMPLE);
//
//    std::vector<cv::Rect> detections;
//
//    for (auto& c : contours) {
//        if (cv::contourArea(c) > 200) {
//            detections.push_back(cv::boundingRect(c));
//        }
//    }
//
//    return detections;
//}
//
//void TrackerManager::associate(
//        std::vector<cv::Rect>& detections) {
//
//    const double MAX_DIST = 120.0;
//
//    std::vector<bool> detectionUsed(detections.size(), false);
//
//    for (auto& tracker : trackers) {
//
//        if (!tracker.valid)
//            continue;
//
//        cv::Point2d prevCenter(
//                tracker.box.x + tracker.box.width / 2,
//                tracker.box.y + tracker.box.height / 2
//        );
//
//        double minDist = 1e9;
//        int bestIdx = -1;
//
//        for (int i = 0; i < detections.size(); i++) {
//
//            if (detectionUsed[i])
//                continue;
//
//            if (tracker.hasConstraint) {
////                if (!(tracker.constraintBox & detections[i]).area())
////                    continue;
//            }
//
//            cv::Point2d center(
//                    detections[i].x + detections[i].width / 2,
//                    detections[i].y + detections[i].height / 2
//            );
//
//            double dist = cv::norm(center - prevCenter);
//
//            if (dist < minDist) {
//                minDist = dist;
//                bestIdx = i;
//            }
//        }
//
//        if (bestIdx != -1 && minDist < MAX_DIST) {
//            tracker.box = detections[bestIdx];
//            tracker.lostFrames = 0;
//            detectionUsed[bestIdx] = true;
//        } else {
//            tracker.lostFrames++;
//            if (tracker.lostFrames > 10)
//                tracker.valid = false;
//        }
//    }
//}
//
//std::vector<TrackerResult> TrackerManager::update(
//        const cv::Mat& frame) {
//
//    auto detections = detectColorBlobs(frame);
//
//    associate(detections);
//
//    std::vector<TrackerResult> results;
//
//    for (auto& t : trackers) {
//        results.push_back({
//                t.id,
//                t.box,
//                t.valid
//        });
//    }
//
//    return results;
//}
