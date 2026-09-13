#include "TrackerManager.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace {

constexpr int kHueRange = 180;           // OpenCV 8-bit hue is in [0, 180)
constexpr double kMinContourArea = 9.0;  // blobs smaller than this (pixels) are noise

int clampByte(int value) {
    return std::clamp(value, 0, 255);
}

}  // namespace

ColorTracker::ColorTracker(int id, HsvColor color, HsvTolerance tolerance)
        : trackerId_(id), color_(color), tolerance_(tolerance) {}

TrackerResult ColorTracker::update(const cv::Mat& hsvFrame, cv::Mat& mask, cv::Mat& mask2) const {
    TrackerResult result;
    result.trackerId = trackerId_;

    const int sLo = clampByte(color_.s - tolerance_.sat);
    const int sHi = clampByte(color_.s + tolerance_.sat);
    const int vLo = clampByte(color_.v - tolerance_.val);
    const int vHi = clampByte(color_.v + tolerance_.val);
    const int hueTol = std::clamp(tolerance_.hue, 0, kHueRange / 2);
    const int hLo = color_.h - hueTol;
    const int hHi = color_.h + hueTol;

    // Hue is circular: a window around red (h near 0 or near 179) has to be split in two,
    // because cv::inRange simply clamps out-of-range bounds and never wraps.
    if (hueTol >= kHueRange / 2) {
        cv::inRange(hsvFrame, cv::Scalar(0, sLo, vLo), cv::Scalar(kHueRange, sHi, vHi), mask);
    } else if (hLo < 0) {
        cv::inRange(hsvFrame, cv::Scalar(0, sLo, vLo), cv::Scalar(hHi, sHi, vHi), mask);
        cv::inRange(hsvFrame, cv::Scalar(kHueRange + hLo, sLo, vLo), cv::Scalar(kHueRange, sHi, vHi), mask2);
        cv::bitwise_or(mask, mask2, mask);
    } else if (hHi >= kHueRange) {
        cv::inRange(hsvFrame, cv::Scalar(hLo, sLo, vLo), cv::Scalar(kHueRange, sHi, vHi), mask);
        cv::inRange(hsvFrame, cv::Scalar(0, sLo, vLo), cv::Scalar(hHi - kHueRange, sHi, vHi), mask2);
        cv::bitwise_or(mask, mask2, mask);
    } else {
        cv::inRange(hsvFrame, cv::Scalar(hLo, sLo, vLo), cv::Scalar(hHi, sHi, vHi), mask);
    }

    // Morphological opening (3x3) removes isolated specks so that sensor noise can never win
    // the "largest blob" selection below.
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::Mat());

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const std::vector<cv::Point>* best = nullptr;
    double bestArea = kMinContourArea;
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area > bestArea) {
            bestArea = area;
            best = &contour;
        }
    }
    if (best == nullptr) {
        return result;
    }

    const cv::Moments moments = cv::moments(*best);
    if (moments.m00 <= 0.0) {
        return result;
    }

    result.found = true;
    result.area = bestArea;
    result.boundingBox = cv::boundingRect(*best);
    result.centroid = cv::Point2d(moments.m10 / moments.m00, moments.m01 / moments.m00);
    return result;
}

void TrackerManager::setTracker(int id, HsvColor color, HsvTolerance tolerance) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::lower_bound(trackers_.begin(), trackers_.end(), id,
                               [](const ColorTracker& tracker, int value) { return tracker.id() < value; });
    if (it != trackers_.end() && it->id() == id) {
        *it = ColorTracker(id, color, tolerance);
    } else {
        trackers_.emplace(it, id, color, tolerance);
    }
}

bool TrackerManager::removeTracker(int id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find_if(trackers_.begin(), trackers_.end(),
                           [id](const ColorTracker& tracker) { return tracker.id() == id; });
    if (it == trackers_.end()) {
        return false;
    }
    trackers_.erase(it);
    return true;
}

void TrackerManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    trackers_.clear();
}

std::vector<int> TrackerManager::trackerIds() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int> ids;
    ids.reserve(trackers_.size());
    for (const auto& tracker : trackers_) {
        ids.push_back(tracker.id());
    }
    return ids;
}

std::vector<TrackerResult> TrackerManager::updateAll(const cv::Mat& rgbaFrame, const cv::Rect* roi) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<TrackerResult> results;
    results.reserve(trackers_.size());
    for (const auto& tracker : trackers_) {
        TrackerResult notFound;
        notFound.trackerId = tracker.id();
        results.push_back(notFound);
    }
    if (trackers_.empty() || rgbaFrame.empty()) {
        return results;
    }
    if (rgbaFrame.type() != CV_8UC4 && rgbaFrame.type() != CV_8UC3) {
        return results;
    }

    cv::Rect region(0, 0, rgbaFrame.cols, rgbaFrame.rows);
    if (roi != nullptr) {
        region &= *roi;
    }
    if (region.width <= 0 || region.height <= 0) {
        return results;
    }

    // One colour conversion per frame, shared by all trackers (the alpha channel is ignored).
    const cv::Mat cropped = rgbaFrame(region);
    cv::cvtColor(cropped, hsv_, cv::COLOR_RGB2HSV);

    for (size_t i = 0; i < trackers_.size(); ++i) {
        TrackerResult r = trackers_[i].update(hsv_, mask_, mask2_);
        if (r.found) {
            r.boundingBox.x += region.x;
            r.boundingBox.y += region.y;
            r.centroid.x += region.x;
            r.centroid.y += region.y;
        }
        results[i] = r;
    }
    return results;
}
