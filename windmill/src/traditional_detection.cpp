/**
 * @file test_net_pnp.cpp
 * @author Clarence Stark (3038736583@qq.com)
 * @brief 用于传统算法检测
 * @version 0.1
 * @date 2025-01-04
 *
 * @copyright Copyright (c) 2025
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <traditional_detection.hpp>

using namespace cv;
using namespace std;
using namespace std::chrono;
// 计算点距
double computeDistance(Point p1, Point p2) {
  return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}
cv::Mat applyYawPerspectiveTransform(const cv::Mat &inputImage,
                                     float yawFactor) {
  // 检查输入图像是否为空
  if (inputImage.empty()) {
    std::cerr << "输入图像为空！" << std::endl;
    return cv::Mat();
  }

  int rows = inputImage.rows;
  int cols = inputImage.cols;

  std::vector<cv::Point2f> pts1 = {cv::Point2f(0, 0), cv::Point2f(cols, 0),
                                   cv::Point2f(0, rows),
                                   cv::Point2f(cols, rows)};

  float horizontalOffset = cols * yawFactor; // 根据输入的因子计算水平偏移量
  std::vector<cv::Point2f> pts2 = {
      cv::Point2f(horizontalOffset, 0), cv::Point2f(cols - horizontalOffset, 0),
      cv::Point2f(horizontalOffset / 2, rows),
      cv::Point2f(cols - horizontalOffset / 2, rows)};

  cv::Mat M = cv::getPerspectiveTransform(pts1, pts2);

  cv::Mat warpedImage;
  cv::warpPerspective(inputImage, warpedImage, M, inputImage.size());

  return warpedImage;
}

// 声明函数
double euclideanDistance(Point p1, Point p2);

vector<Point> findIntersectionsByEquation(const Point &center1,
                                          const Point &center2, double radius,
                                          const RotatedRect &ellipse, Mat &pic,
                                          bool drawPoints, WMBlade &blade);

vector<Point> findIntersectionsByContour(const vector<Point> &contour,
                                         const Point &center1,
                                         const Point &center2, Mat &pic,
                                         bool drawPoints);

// 在文件开头添加全局变量和窗口名称
const string WINDOW_NAME = "Parameter Controls";
int circularityThreshold = 50; // 圆度阈值

bool useChannelMinus = true; // 是否使用通道减法提取红色（建议开启，光线干扰小，效果稳定调参简单）
bool useTrackbars = true; // 是否使用滑动条动态调参
bool debug = false;        // debug模式  //输出识别圆个数
int lowH1 = 0;            // 第一个红色范围的低阈值
int highH1 = 65;          // 第一个红色范围的高阈值
int lowH2 = 160;          // 第二个红色范围的低阈值
int highH2 = 179;         // 第二个红色范围的高阈值

int lowS = 160;  // 饱和度下限
int highS = 255; // 饱和度上限
int lowV = 50;   // 亮度下限
int highV = 255; // 亮度上限

int dilationSize = 11;     // 膨胀核大小
int erosionSize = 3;      // 腐蚀核大小
int thresholdValue = 13; // 二值化阈值

int rect_area_threshold = 500;   // 矩形面积阈值
int circle_area_threshold = 200; // 类圆轮廓面积阈值

int length_width_ratio_threshold = 3; // 长宽比阈值

// 添加滑动条回调函数
void onTrackbarCircularity(int value, void *) {
  circularityThreshold = value;
  // cout << "圆度阈值更新为: " << circularityThreshold << endl;
}

void onTrackbarHSV(int value, void *) {
  // 确保阈值范围合理
  if (highH1 <= lowH1)
    highH1 = lowH1 + 1;
  if (highH2 <= lowH2)
    highH2 = lowH2 + 1;

  // cout << "HSV范围更新为: H1(" << lowH1 << "-" << highH1 << "), H2(" << lowH2
  //      << "-" << highH2 << ")" << endl;
}

// 添加创建滑动条
void createTrackbars() {
  namedWindow(WINDOW_NAME, WINDOW_NORMAL);

  // 设置窗口位置到左上角
  moveWindow(WINDOW_NAME, 0, 0);

  // 创建圆度阈值的滑动条 (0-100)
  createTrackbar("Circularity", WINDOW_NAME, &circularityThreshold, 100,
                 onTrackbarCircularity);
  createTrackbar("Dilation Size", WINDOW_NAME, &dilationSize, 21, nullptr);
  createTrackbar("Erosion Size", WINDOW_NAME, &erosionSize, 21, nullptr);
  createTrackbar("Rect Area Threshold", WINDOW_NAME, &rect_area_threshold, 1000,
                 nullptr);
  createTrackbar("Circle Area Threshold", WINDOW_NAME, &circle_area_threshold,
                 1000, nullptr);
  createTrackbar("Length Width Ratio Threshold", WINDOW_NAME,
                 &length_width_ratio_threshold, 10, nullptr);
  // 创建HSV阈值的滑动条
  if (!useTrackbars) {
    createTrackbar("Low H1", WINDOW_NAME, &lowH1, 179, onTrackbarHSV);
    createTrackbar("High H1", WINDOW_NAME, &highH1, 179, onTrackbarHSV);
    createTrackbar("Low H2", WINDOW_NAME, &lowH2, 179, onTrackbarHSV);
    createTrackbar("High H2", WINDOW_NAME, &highH2, 179, onTrackbarHSV);

    createTrackbar("Low S", WINDOW_NAME, &lowS, 255, onTrackbarHSV);
    createTrackbar("High S", WINDOW_NAME, &highS, 255, onTrackbarHSV);
    createTrackbar("Low V", WINDOW_NAME, &lowV, 255, onTrackbarHSV);
    createTrackbar("High V", WINDOW_NAME, &highV, 255, onTrackbarHSV);
  } else {
    createTrackbar("Threshold", WINDOW_NAME, &thresholdValue, 255, nullptr);
  }
}

/**
 * @brief 图像预处理函数
 * @param inputImage 输入图像
 * @param useChannelMinus 是否使用通道相减法
 * @param debug 是否显示中间步骤
 * @return 预处理后的掩码图像
 */
cv::Mat preprocess(const cv::Mat &inputImage, bool useChannelMinus,
                   bool debug) {
  cv::Mat final_mask;

  if (!useChannelMinus) {
    // 1. HSV转换和掩码处理阶段
    cv::Mat processedImage;
    GaussianBlur(inputImage, processedImage, Size(3, 3), 0);

    cv::Mat hsv;
    cvtColor(processedImage, hsv, COLOR_BGR2HSV);

    // 定义红色的HSV范围
    cv::Mat mask1, mask2;
    inRange(hsv, Scalar(lowH1, lowS, lowV), Scalar(highH1, highS, highV),
            mask1);
    inRange(hsv, Scalar(lowH2, lowS, lowV), Scalar(highH2, highS, highV),
            mask2);

    cv::Mat red_mask;
    addWeighted(mask1, 1.0, mask2, 1.0, 0.0, red_mask);

    Mat kernel1 =
        getStructuringElement(MORPH_RECT, Size(dilationSize, dilationSize));
    Mat kernel2 =
        getStructuringElement(MORPH_RECT, Size(erosionSize, erosionSize));

    cv::Mat dilated_mask;
    dilate(red_mask, dilated_mask, kernel1);
    erode(dilated_mask, final_mask, kernel2);
  } else {
    std::vector<cv::Mat> channels;
    cv::split(inputImage, channels);

    cv::Mat blue = channels[0];
    cv::Mat red = channels[2];

    // 通道相减得到灰度图
    Mat temp;
    subtract(red, blue, temp);
    if (debug) {
      imshow("red_part_binary", temp);
    }

    // 对灰度图进行二值化
    threshold(temp, final_mask, thresholdValue, 255, THRESH_BINARY);
    if (debug) {
      imshow("red_part_threshold", final_mask);
    }

    Mat kernel1 =
        getStructuringElement(MORPH_RECT, Size(dilationSize, dilationSize));
    Mat kernel2 =
        getStructuringElement(MORPH_RECT, Size(erosionSize, erosionSize));

    dilate(final_mask, final_mask, kernel1);
    if (debug) {
      imshow("red_part_dilate", final_mask);
    }
    erode(final_mask, final_mask, kernel2);
    if (debug) {
      imshow("red_part_erode", final_mask);
    }
  }

  return final_mask;
}

/**
 * @brief 关键点结构体
 * @param circleContours 类圆轮廓
 * @param circleAreas 类圆轮廓面积
 * @param circularities 类圆轮廓圆度
 * @param rectCenter 矩形中心点
 * @param circlePoints 类圆轮廓中心点
 */

/**
 * @brief 检测关键点
 * @param contours 轮廓
 * @param hierarchy 轮廓层次
 * @param processedImage 处理后的图像
 * @return 关键点结果
 */

KeyPoints detect_key_points(const vector<vector<Point>> &contours,
                            const vector<Vec4i> &hierarchy, Mat &processedImage,
                            WMBlade &blade) {
  KeyPoints result;

  // 处理每个轮廓
  for (int i = 0; i < contours.size(); i++) {
    const auto &contour = contours[i];
    // 检查是否为子轮廓
    if (hierarchy[i][3] != -1) { // 如果有父轮廓，则跳过
      continue;
    }
    double area = contourArea(contour);
    if (area > rect_area_threshold) {
      // 计算最小外接矩形
      RotatedRect rect = minAreaRect(contour);
      float width = rect.size.width;
      float height = rect.size.height;

      // 确保width是较长的边
      float aspectRatio = width > height ? width / height : height / width;

      // 如果长宽比大于设定阈值，则认为是流水灯条（此处逻辑可以改进，使用容器存储所有满足条件的轮廓，然后排序选取那个面积最大的轮廓，而不仅仅是直接确定rectCenter，虽然绝大多数情况下都只会有流水灯条满足条件，但这样可以更加鲁棒）
      if (aspectRatio > length_width_ratio_threshold) {
        // 绘制矩形
        Point2f vertices[4];
        rect.points(vertices);
        if (debug) {
          for (int j = 0; j < 4; j++) {
            line(processedImage, vertices[j], vertices[(j + 1) % 4],
                 Scalar(0, 255, 0), 2);
          }
        }

        // 使用轮廓的质心代替矩形中心
        Moments m = moments(contour);
        result.rectCenter = Point(m.m10 / m.m00, m.m01 / m.m00);

        // 绘制质心
        circle(processedImage, result.rectCenter, 3, Scalar(0, 255, 0), -1);     //流水灯中心
      }
    }

    if (area > circle_area_threshold) {
      double circularity =
          4 * CV_PI * area / (pow(arcLength(contour, true), 2));

      //
      if (circularity > (circularityThreshold / 100.0)) {
        // 计算子轮廓数量
        int childCount = 0;
        int firstChild = hierarchy[i][2];
        if (firstChild >= 0) {
          childCount = 1;                           // 至少有一个子轮廓
          int nextChild = hierarchy[firstChild][0]; // 下一个同级轮廓
          while (nextChild >= 0 && nextChild != firstChild) {
            childCount++;
            nextChild = hierarchy[nextChild][0];
          }
        }
        if (childCount > 1 || childCount == 0 && hierarchy[i][3] == -1) {
          result.circleContours.push_back(contour);
          result.circleAreas.push_back(area);
          result.circularities.push_back(circularity);

          Moments m = moments(contour);
          Point circleCenter(int(m.m10 / m.m00), int(m.m01 / m.m00));
          result.circlePoints.push_back(circleCenter);
        }
      }
    }
  }
  if (debug) {
    cout << "检测到 " << result.circleContours.size()
         << " 个类圆轮廓 (阈值: " << circularityThreshold << ")" << endl;
  }
  return result;
}
DetectionResult detect(const cv::Mat &inputImage, bool useEquationMethod,
                       bool debug, WMBlade &blade) {
  if (useTrackbars) {
    static bool trackbarsInitialized = false;
    if (!trackbarsInitialized) {
      createTrackbars();
      trackbarsInitialized = true;
    }
  }

  DetectionResult result;
  auto start_time = high_resolution_clock::now();

  // 预处理
  Mat final_mask = preprocess(inputImage, useChannelMinus, debug);

  // 轮廓分析阶段,提取能量机关扇叶中心点，流水灯条中心点以及R标
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(final_mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

  // 创建输入图像的副本
  Mat processedImage = inputImage.clone();

  KeyPoints keyPoints =
      detect_key_points(contours, hierarchy, processedImage, blade);

  // 绘制类圆轮廓
  if (debug) {
    Mat contourImage;
    cvtColor(final_mask, contourImage, COLOR_GRAY2BGR);
    for (const auto &contour : keyPoints.circleContours) {
      drawContours(contourImage, vector<vector<Point>>{contour}, 0,
                   Scalar(128, 128, 128), 2);
    }
    imshow("circle_contours", contourImage);
  }

  // 按面积从大到小排序类圆轮廓
  vector<size_t> indices(keyPoints.circleContours.size());
  iota(indices.begin(), indices.end(), 0);
  sort(indices.begin(), indices.end(), [&keyPoints](size_t i1, size_t i2) {
    return keyPoints.circleAreas[i1] > keyPoints.circleAreas[i2];
  });

  // 交点计算
  if (keyPoints.circleContours.size() >= 2) {
    blade.apex.push_back(keyPoints.circlePoints[indices[1]]);    //放进R点
    blade.apex.push_back(keyPoints.circlePoints[indices[0]]);
    Moments m1 = moments(keyPoints.circleContours[indices[0]]);
    // 拟合椭圆轮廓
    cv::RotatedRect ellipse =
        cv::fitEllipse(keyPoints.circleContours[indices[0]]);
    // 使用椭圆中心代替矩计算的中心点
    Point center1(ellipse.center.x, ellipse.center.y);
    double radius = sqrt(keyPoints.circleAreas[indices[0]] / CV_PI);

    if (useEquationMethod) {
      result.intersections =
          findIntersectionsByEquation(center1, keyPoints.rectCenter, radius,
                                      ellipse, processedImage, debug, blade);
    } else {
      result.intersections = findIntersectionsByContour(
          keyPoints.circleContours[indices[0]], center1, keyPoints.rectCenter,
          processedImage, debug);
    }
  }
  result.processedImage = processedImage; // 处理后的图像

  if (debug) {
    for (const auto &point : keyPoints.circlePoints) {
      circle(result.processedImage, point, 3, Scalar(0, 255, 0), -1);
    }
    cv::imshow("Processed Frame", result.processedImage);
  }

  // 计算处理时间
  auto end_time = high_resolution_clock::now();
  result.processingTime =
      duration_cast<milliseconds>(end_time - start_time).count();
  blade.apex.push_back(keyPoints.rectCenter);

  return result;
}

double euclideanDistance(Point p1, Point p2) {
  return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// 通过方程求解交点的方法
vector<Point> findIntersectionsByEquation(const Point &center1,
                                          const Point &center2, double radius,
                                          const RotatedRect &ellipse, Mat &pic,
                                          bool drawPoints, WMBlade &blade) {
  vector<Point> intersections;

  // 获取椭圆参数
  Point2f ellipse_center = ellipse.center;
  Size2f size = ellipse.size;
  float angle_deg = ellipse.angle;              // 旋转角度（度）
  double angle_rad = angle_deg * CV_PI / 180.0; // 旋转角度（弧度）

  // 将椭圆缩放0.9倍
  double scale = 0.95;
  // 半长轴和半短轴缩放
  double a = (size.width / 2.0) * scale;
  double b = (size.height / 2.0) * scale;

  // 如果需要在图像上显示缩放后的椭圆（用于调试）
  if (drawPoints) {
    RotatedRect scaled_ellipse(ellipse_center,
                               Size2f(size.width * scale, size.height * scale),
                               angle_deg);
    if (debug) {
      // 绘制椭圆
      cv::ellipse(pic, scaled_ellipse, cv::Scalar(0, 255, 0), 2);

      // 计算长轴和短轴的端点
      double cos_angle = cos(angle_rad);
      double sin_angle = sin(angle_rad);

      // 长轴端点
      Point2f major_axis_p1(ellipse_center.x + (a * cos_angle),
                            ellipse_center.y + (a * sin_angle));
      Point2f major_axis_p2(ellipse_center.x - (a * cos_angle),
                            ellipse_center.y - (a * sin_angle));

      // 短轴端点 (垂直于长轴)
      Point2f minor_axis_p1(ellipse_center.x - (b * sin_angle),
                            ellipse_center.y + (b * cos_angle));
      Point2f minor_axis_p2(ellipse_center.x + (b * sin_angle),
                            ellipse_center.y - (b * cos_angle));

      // 绘制长轴和短轴
      // line(pic, major_axis_p1, major_axis_p2, cv::Scalar(0, 0, 255),
      //      2); // 红色长轴
      // line(pic, minor_axis_p1, minor_axis_p2, cv::Scalar(255, 0, 0),
      //      2); // 蓝色短轴
    }
  }

  // 计算第一条直线的系数 A x + B y + C = 0
  double A = center2.y - center1.y;
  double B = center1.x - center2.x;
  double C = center2.x * center1.y - center1.x * center2.y;

  // 将直线方程旋转到椭圆的坐标系
  double cos_theta = cos(angle_rad);
  double sin_theta = sin(angle_rad);

  double A_rot = A * cos_theta + B * sin_theta;
  double B_rot = -A * sin_theta + B * cos_theta;
  double C_rot = C + A * ellipse_center.x + B * ellipse_center.y;

  // 避免除零的情况
  if (fabs(B_rot) < 1e-8) {
    // 如果需要，可添加额外的逻辑来处理这种直线几乎垂直的情况，但测了一下貌似影响不是很大，最多也就一两帧，如果后续有问题这里可以考虑优化此处。
  }

  // 计算二次方程系数
  double M = (1.0 / (a * a)) + (A_rot * A_rot) / (B_rot * B_rot * b * b);
  double N = (2.0 * A_rot * C_rot) / (B_rot * B_rot * b * b);
  double P = (C_rot * C_rot) / (B_rot * B_rot * b * b) - 1.0;

  // delta
  double discriminant = N * N - 4.0 * M * P;

  if (discriminant >= 0) {
    double sqrt_discriminant = sqrt(discriminant);
    double x1_rot = (-N + sqrt_discriminant) / (2.0 * M);
    double x2_rot = (-N - sqrt_discriminant) / (2.0 * M);

    double y1_rot = (-A_rot * x1_rot - C_rot) / B_rot;
    double y2_rot = (-A_rot * x2_rot - C_rot) / B_rot;

    double x1 = x1_rot * cos_theta - y1_rot * sin_theta + ellipse_center.x;
    double y1 = x1_rot * sin_theta + y1_rot * cos_theta + ellipse_center.y;

    double x2 = x2_rot * cos_theta - y2_rot * sin_theta + ellipse_center.x;
    double y2 = x2_rot * sin_theta + y2_rot * cos_theta + ellipse_center.y;

    Point pt1(cvRound(x1), cvRound(y1));
    Point pt2(cvRound(x2), cvRound(y2));
    if (computeDistance(pt1, center2) > computeDistance(pt2, center2)) {
      intersections.emplace_back(pt1);
      blade.apex.push_back(pt1);
    } else {
      intersections.emplace_back(pt2);
      blade.apex.push_back(pt2);
    }
    circle(pic, intersections[0], 3, Scalar(0, 255, 0), -1);    //扇叶顶端点

    if (drawPoints) {

      // circle(pic, intersections[1], 2, Scalar(0, 255, 0), -1);
      line(pic, center2, intersections[0], Scalar(255, 0, 0), 2);
    }
  }

  // 共轭直径
  if (A != 0) { // 确保A不为零以避免除以零
    double new_slope = (b * b * B) / (a * a * A);

    double A2 = new_slope;
    double B2 = -1.0;
    double C2 = center1.y - new_slope * center1.x;

    double A2_rot = A2 * cos_theta + B2 * sin_theta;
    double B2_rot = -A2 * sin_theta + B2 * cos_theta;
    double C2_rot = C2 + A2 * ellipse_center.x + B2 * ellipse_center.y;

    double M2 = (1.0 / (a * a)) + (A2_rot * A2_rot) / (B2_rot * B2_rot * b * b);
    double N2 = (2.0 * A2_rot * C2_rot) / (B2_rot * B2_rot * b * b);
    double P2 = (C2_rot * C2_rot) / (B2_rot * B2_rot * b * b) - 1.0;

    double discriminant2 = N2 * N2 - 4.0 * M2 * P2;

    if (discriminant2 >= 0) {
      double sqrt_discriminant2 = sqrt(discriminant2);
      double x1_rot2 = (-N2 + sqrt_discriminant2) / (2.0 * M2);
      double x2_rot2 = (-N2 - sqrt_discriminant2) / (2.0 * M2);

      double y1_rot2 = (-A2_rot * x1_rot2 - C2_rot) / B2_rot;
      double y2_rot2 = (-A2_rot * x2_rot2 - C2_rot) / B2_rot;

      double x1_2 =
          x1_rot2 * cos_theta - y1_rot2 * sin_theta + ellipse_center.x;
      double y1_2 =
          x1_rot2 * sin_theta + y1_rot2 * cos_theta + ellipse_center.y;

      double x2_2 =
          x2_rot2 * cos_theta - y2_rot2 * sin_theta + ellipse_center.x;
      double y2_2 =
          x2_rot2 * sin_theta + y2_rot2 * cos_theta + ellipse_center.y;

      Point pt3(cvRound(x1_2), cvRound(y1_2));
      Point pt4(cvRound(x2_2), cvRound(y2_2));

      intersections.emplace_back(pt3);
      intersections.emplace_back(pt4);
      circle(pic, pt3, 3, Scalar(0, 255, 0), -1);   //扇叶三号点
      circle(pic, pt4, 3, Scalar(0, 255, 0), -1);
      // 排序左右两个关键点的算法
      Point O = blade.apex[0]; // 获取O点
      Point OP3 = pt3 - O;     // OP3向量
      Point OP4 = pt4 - O;     // OP4向量

      // 计算OP3N = (-OP3_y, OP3_x)
      Point OP3N(-OP3.y, OP3.x);

      // 计算OP3N和OP4的点积
      double dotProduct = OP3N.x * OP4.x + OP3N.y * OP4.y;

      // 根据点积符号决定push顺序
      if (dotProduct < 0) {
        intersections.emplace_back(pt3);
        blade.apex.push_back(pt3);
        intersections.emplace_back(pt4);
        blade.apex.push_back(pt4);
      } else {
        intersections.emplace_back(pt4);
        blade.apex.push_back(pt4);
        intersections.emplace_back(pt3);
        blade.apex.push_back(pt3);
      }

      if (drawPoints) {

        line(pic, pt3, pt4, Scalar(255, 0, 0), 2);

        Point line_pt1, line_pt2;
        if (fabs(new_slope) < 1e-8) {
          // 斜率几乎为零，水平线
          line_pt1 = Point(0, cvRound(center1.y));
          line_pt2 = Point(pic.cols, cvRound(center1.y));
        } else {
          double y_start = new_slope * (0 - center1.x) + center1.y;
          double y_end = new_slope * (pic.cols - center1.x) + center1.y;
          line_pt1 = Point(0, cvRound(y_start));
          line_pt2 = Point(pic.cols, cvRound(y_end));
        }
      }
    }
  }
  return intersections;
}
// 通过轮廓点求解交点的方法（已弃用）
vector<Point> findIntersectionsByContour(const vector<Point> &contour,
                                         const Point &center1,
                                         const Point &center2, Mat &pic,
                                         bool drawPoints = true) {
  double a = center2.y - center1.y;
  double b = center1.x - center2.x;
  double c = center2.x * center1.y - center1.x * center2.y;

  vector<Point> intersections;
  vector<pair<double, Point>> distances; // 存储所有点到直线的距离及点的坐标

  // 计算所有轮廓点到直线的距离
  for (const Point &p : contour) {
    double dist = abs(a * p.x + b * p.y + c) / sqrt(a * a + b * b);
    distances.push_back({dist, p});
  }

  // 第一次排序（直线交点）
  sort(distances.begin(), distances.end(),
       [](const pair<double, Point> &a, const pair<double, Point> &b) {
         return a.first < b.first;
       });

  // 从最近的N个点中找到方向最接近相反的一对点
  const int N = min(10, (int)distances.size()); // 考虑距离最近的10个点
  Point bestPoint1, bestPoint2;
  double bestAngleDiff = 0;
  double minDist = DBL_MAX;

  for (int i = 0; i < N; i++) {
    for (int j = i + 1; j < N; j++) {
      Point p1 = distances[i].second;
      Point p2 = distances[j].second;

      // 计算两个点到圆心的向量
      Point vec1 = p1 - center1;
      Point vec2 = p2 - center1;

      // 计算向量的点积
      double dotProduct = vec1.x * vec2.x + vec1.y * vec2.y;
      double mag1 = sqrt(vec1.x * vec1.x + vec1.y * vec1.y);
      double mag2 = sqrt(vec2.x * vec2.x + vec2.y * vec2.y);

      // 计算夹角的余弦值
      double cosTheta = dotProduct / (mag1 * mag2);

      // 希望夹角接近180度，即余弦值接近-1
      if (cosTheta < -0.9) { // 夹角几乎为平角
        double totalDist = distances[i].first + distances[j].first;
        if (totalDist < minDist) {
          minDist = totalDist;
          bestPoint1 = p1;
          bestPoint2 = p2;
          bestAngleDiff = acos(cosTheta) * 180 / CV_PI;
        }
      }
    }
  }

  // 找到了合适的点对
  if (minDist != DBL_MAX) {
    intersections.push_back(bestPoint1);
    intersections.push_back(bestPoint2);

    if (drawPoints) {
      circle(pic, bestPoint1, 2, Scalar(0, 255, 0), -1);
      circle(pic, bestPoint2, 2, Scalar(0, 255, 0), -1);
      line(pic, center1, bestPoint1, Scalar(0, 255, 0), 1);
      line(pic, center1, bestPoint2, Scalar(0, 255, 0), 1);
    }

    cout << "找到的交点夹角: " << bestAngleDiff << " 度" << endl;
  }

  // 对垂直线重复相同的过程
  double perpendicular_a = -b;
  double perpendicular_b = a;
  double perpendicular_c =
      -(perpendicular_a * center1.x + perpendicular_b * center1.y);

  distances.clear();
  for (const Point &p : contour) {
    double dist =
        abs(perpendicular_a * p.x + perpendicular_b * p.y + perpendicular_c) /
        sqrt(perpendicular_a * perpendicular_a +
             perpendicular_b * perpendicular_b);
    distances.push_back({dist, p});
  }

  // 第二次排序（垂直线交点）
  sort(distances.begin(), distances.end(),
       [](const pair<double, Point> &a, const pair<double, Point> &b) {
         return a.first < b.first;
       });

  bestAngleDiff = 0;
  minDist = DBL_MAX;

  for (int i = 0; i < N; i++) {
    for (int j = i + 1; j < N; j++) {
      Point p1 = distances[i].second;
      Point p2 = distances[j].second;

      Point vec1 = p1 - center1;
      Point vec2 = p2 - center1;

      double dotProduct = vec1.x * vec2.x + vec1.y * vec2.y;
      double mag1 = sqrt(vec1.x * vec1.x + vec1.y * vec1.y);
      double mag2 = sqrt(vec2.x * vec2.x + vec2.y * vec2.y);

      double cosTheta = dotProduct / (mag1 * mag2);

      if (cosTheta < -0.8) {
        double totalDist = distances[i].first + distances[j].first;
        if (totalDist < minDist) {
          minDist = totalDist;
          bestPoint1 = p1;
          bestPoint2 = p2;
          bestAngleDiff = acos(cosTheta) * 180 / CV_PI;
        }
      }
    }
  }

  if (minDist != DBL_MAX) {
    intersections.push_back(bestPoint1);
    intersections.push_back(bestPoint2);

    if (drawPoints) {
      circle(pic, bestPoint1, 2, Scalar(0, 255, 0), -1);
      circle(pic, bestPoint2, 2, Scalar(0, 255, 0), -1);
      line(pic, center1, bestPoint1, Scalar(0, 255, 0), 1);
      line(pic, center1, bestPoint2, Scalar(0, 255, 0), 1);
    }

    cout << "找到的垂直线交点夹角: " << bestAngleDiff << " 度" << endl;
  }

  return intersections;
}

// int main() {
//   cout << "传统算法检测" << endl;

//   bool useOffcialWindmill = true;
//   bool perspective = true;
//   bool useVideo = true;

//   if (useVideo) {
//     VideoCapture cap;
//     if (!useOffcialWindmill) {
//       cap = VideoCapture("/Users/clarencestark/RoboMaster/第四次任务/"
//                          "nanodet_rm/camera/build/output.avi");
//     } else {
//       // cap =
//       // VideoCapture("/Users/clarencestark/RoboMaster/步兵打符-视觉组/"
//       // "传统识别算法/local_Indentify_Develop/src/output.mp4");
//       // cap = VideoCapture(
//       //     "/Users/clarencestark/RoboMaster/步兵打符-视觉组/"
//       //     "传统识别算法/XJTU2025WindMill/imgs_and_videos/output.mp4");
//       cap = VideoCapture("/Users/clarencestark/RoboMaster/第四次任务/"
//                          "nanodet_rm/camera/build/output222.mp4");
//     }

//     if (!cap.isOpened()) {
//       cout << "无法打开视频文件" << endl;
//       return -1;
//     }

//     Mat frame;
//     bool paused = false;

//     // 添加帧率计算相关变量
//     double fps = 0;
//     auto last_time = high_resolution_clock::now();
//     int frame_count = 0;

//     while (true) {
//       if (!paused) {
//         if (!cap.read(frame)) {
//           cout << "视频结束或帧获取失败" << endl;
//           break;
//         }

//         // 计算帧率
//         frame_count++;
//         auto current_time = high_resolution_clock::now();
//         auto time_diff =
//             duration_cast<milliseconds>(current_time - last_time).count();

//         if (time_diff >= 1000) { // 每秒更新一次帧率
//           fps = frame_count * 1000.0 / time_diff;
//           frame_count = 0;
//           last_time = current_time;
//         }
//       }

//       DetectionResult result;
//       if (perspective) {
//         Mat transformedFrame = applyYawPerspectiveTransform(frame, 0.18);
//         WMBlade temp_blade;
//         result = detect(transformedFrame, true, debug, temp_blade);
//         imshow("Original Image", frame);
//         imshow("Transformed Image", transformedFrame);
//       } else {
//         WMBlade temp_blade;
//         result = detect(frame, true, debug, temp_blade);
//       }

//       // 帧率和处理时间
//       string fps_text = "FPS: " + to_string(static_cast<int>(fps));
//       string time_text =
//           "Process Time: " + to_string(result.processingTime) + "ms";
//       putText(result.processedImage, fps_text, Point(10, 30),
//               FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
//       putText(result.processedImage, time_text, Point(10, 70),
//               FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

//       // 显示结果图像
//       imshow("Processed Image", result.processedImage);

//       // 等待按键
//       char key = (char)waitKey(70);

//       // 按键控制
//       if (key == 'q' || key == 'Q') {
//         break; // 退出
//       } else if (key == ' ') {
//         paused = !paused; // 空格键切换暂停/继续
//       }
//     }

//     cap.release();
//   } else {
//     // 原有的图像处理逻辑
//     Mat frame;
//     if (!useOffcialWindmill) {
//       frame = imread("/Users/clarencestark/RoboMaster/第四次任务/nanodet_rm/"
//                      "camera/build/imgs/image52.jpg");
//     } else {
//       frame = imread("/Users/clarencestark/RoboMaster/步兵打符-视觉组/"
//                      "local_Indentify_Develop/src/test3.jpg");
//     }

//     if (frame.empty()) {
//       cout << "无法获取图像" << endl;
//       return -1;
//     }

//     WMBlade temp_blade;
//     DetectionResult result;
//     if (perspective) {
//       Mat transformedFrame = applyYawPerspectiveTransform(frame, 0.20);

//       // 显示原始图像和变换后的图像
//       imshow("Original Image", frame);
//       imshow("Transformed Image", transformedFrame);

//       // 处理变换后的图像
//       result = detect(transformedFrame, true, false, temp_blade);
//     } else {
//       result = detect(frame, true, false, temp_blade);
//     }

//     // 显示结果
//     cout << "处理时间: " << result.processingTime << " ms" << endl;
//     cout << "检测到 " << result.circlePoints.size() << " 个圆心" << endl;
//     cout << "检测到 " << result.intersections.size() << " 个交点" << endl;

//     // 显示结果图像
//     imshow("Processed Image", result.processedImage);
//     waitKey(0);
//   }

//   return 0;
// }
