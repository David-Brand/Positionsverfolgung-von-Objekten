#ifndef TRACKINGPLOT_TRACKERMANAGER_HPP
#define TRACKINGPLOT_TRACKERMANAGER_HPP

#include <opencv2/core.hpp>

#include <mutex>
#include <vector>

/** Colour in OpenCV's 8-bit HSV units: h in [0, 180), s and v in [0, 255]. */
struct HsvColor {
    int h;
    int s;
    int v;
};

/** Half-widths of the accepted window around an HsvColor, in the same units. */
struct HsvTolerance {
    int hue;
    int sat;
    int val;
};

struct TrackerResult {
    int trackerId = -1;
    bool found = false;
    cv::Rect boundingBox;   // pixel coordinates in the analysed frame
    cv::Point2d centroid;   // pixel coordinates (sub-pixel accurate)
    double area = 0.0;      // area of the tracked blob in pixels
};

/** Segments one colour and returns the largest matching blob. */
class ColorTracker {
public:
    ColorTracker(int id, HsvColor color, HsvTolerance tolerance);

    int id() const { return trackerId_; }

    /**
     * @param hsvFrame CV_8UC3 image in HSV (H in [0,180)).
     * @param mask, mask2 scratch buffers that are reused between calls.
     */
    TrackerResult update(const cv::Mat& hsvFrame, cv::Mat& mask, cv::Mat& mask2) const;

private:
    int trackerId_;
    HsvColor color_;
    HsvTolerance tolerance_;
};

/**
 * Owns a set of colour trackers. All methods are thread-safe: the camera analysis thread calls
 * updateAll() while the UI thread may add, replace or remove trackers.
 */
class TrackerManager {
public:
    void setTracker(int id, HsvColor color, HsvTolerance tolerance);
    bool removeTracker(int id);
    void clear();
    std::vector<int> trackerIds();

    /**
     * Runs every tracker on the frame.
     * @param rgbaFrame CV_8UC4 (R,G,B,A) or CV_8UC3 (R,G,B) image.
     * @param roi optional pixel rectangle; only this part of the frame is analysed. Results are
     *            still expressed in full-frame pixel coordinates.
     * @return one result per tracker, ordered by tracker id.
     */
    std::vector<TrackerResult> updateAll(const cv::Mat& rgbaFrame, const cv::Rect* roi);

private:
    std::mutex mutex_;
    std::vector<ColorTracker> trackers_;  // kept sorted by id
    cv::Mat hsv_;
    cv::Mat mask_;
    cv::Mat mask2_;
};

#endif  // TRACKINGPLOT_TRACKERMANAGER_HPP
