#ifndef TRADITIONAL_DETECTION_HPP
#define TRADITIONAL_DETECTION_HPP

#include "WMIdentify.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

struct DetectionResult {
  std::vector<cv::Point> intersections;
  std::vector<cv::Point> circlePoints;
  cv::Mat processedImage;
  double processingTime;
};

struct KeyPoints {
  std::vector<std::vector<cv::Point>> circleContours;
  std::vector<double> circleAreas;
  std::vector<double> circularities;
  cv::Point2f rectCenter;
  std::vector<cv::Point> circlePoints;
};

DetectionResult detect(const cv::Mat &inputImage, bool useEquationMethod,
                       bool debug, WMBlade &blade);

KeyPoints detect_key_points(const std::vector<std::vector<cv::Point>> &contours,
                            const std::vector<cv::Vec4i> &hierarchy,
                            cv::Mat &processedImage, WMBlade &blade);

std::vector<cv::Point>
findIntersectionsByEquation(const cv::Point &center1, const cv::Point &center2,
                            double radius, const cv::RotatedRect &ellipse,
                            cv::Mat &pic, bool drawPoints, WMBlade &blade);

#endif // TRADITIONAL_DETECTION_HPP
