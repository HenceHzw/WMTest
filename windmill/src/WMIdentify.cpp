/**
 * @file WMIdentify.cpp
 * @author Clarence Stark (3038736583@qq.com)
 * @brief 任意点位打符识别类实现
 * @version 0.1
 * @date 2024-12-08
 *
 * @copyright Copyright (c) 2024
 */

#include "WMIdentify.hpp"
#include "globalParam.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "traditional_detection.hpp"
#include <algorithm>
#include <chrono>
#include <deque>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <ostream>
// #define Pi 3.1415926
const double Pi = 3.1415926;

#ifndef ON
#define ON 1
#endif

#ifndef OFF
#define OFF 0
#endif

/**
 * @brief WMIdentify类构造函数
 * @param[in] gp     全局参数
 * @return void
 */
WMIdentify::WMIdentify(GlobalParam &gp) {
  this->gp = &gp;
  this->t_start =
      std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
  // 从gp中读取一些数据
  this->switch_INFO = this->gp->switch_INFO;
  this->switch_ERROR = this->gp->switch_ERROR;
  // this->get_armor_mode = this->gp->get_armor_mode;
  // 将R与Wing的状态设置为没有读取到内容
  this->R_stat = 0;
  this->Wing_stat = 0;
  this->Winghat_stat = 0;
  this->R_estimate.x = 0;
  this->R_estimate.y = 0;
  // this->d_RP2 = 0.7; // ❗️大符半径？
  // this->d_RP1P3 = 0.6;
  // this->d_P1P3 = 0.2;
  this->camera_matrix = this->gp->camera_matrix;
  this->dist_coeffs = this->gp->dist_coeffs;
  this->data_img = cv::Mat::zeros(400, 800, CV_8UC3);
  // 输出日志，初始化成功
  // LOG_IF(INFO, this->switch_INFO) << "WMIdentify Successful";
}

/**
 * @brief WMIdentify类析构函数
 * @return void
 */
WMIdentify::~WMIdentify() {
  // WMIdentify之中的内容都会自动析构
  // std::cout << "析构中，下次再见喵～" << std::endl;
  // 输出日志，析构成功
  // LOG_IF(INFO, this->switch_INFO) << "~WMIdentify Successful";
}

void drawFixedWorldAxes(cv::Mat &frame, const cv::Mat &cameraMatrix,
                        const cv::Mat &distCoeffs, const cv::Mat &R_init,
                        const cv::Mat &t_init) {
  // 定义世界坐标系中的坐标轴点
  std::vector<cv::Point3f> axisPoints;
  axisPoints.push_back(cv::Point3f(0, 0, 0));   // 原点
  axisPoints.push_back(cv::Point3f(0.1, 0, 0)); // X轴端点
  axisPoints.push_back(cv::Point3f(0, 0.1, 0)); // Y轴端点
  axisPoints.push_back(cv::Point3f(0, 0, 0.1)); // Z轴端点

  // 计算旋转向量和平移向量
  cv::Mat rvec, tvec;
  cv::Rodrigues(R_init, rvec); // 将旋转矩阵转换为旋转向量
  tvec = t_init.clone();

  // 将3D点投影到图像平面
  std::vector<cv::Point2f> imagePoints;
  cv::projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs,
                    imagePoints);

  // 绘制坐标轴
  cv::line(frame, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255),
           2); // X轴-红色
  cv::line(frame, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0),
           2); // Y轴-绿色
  cv::line(frame, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0),
           2); // Z轴-蓝色

  // 添加坐标轴标签
  cv::putText(frame, "X", imagePoints[1], cv::FONT_HERSHEY_SIMPLEX, 1.0,
              cv::Scalar(0, 0, 255), 2);
  cv::putText(frame, "Y", imagePoints[2], cv::FONT_HERSHEY_SIMPLEX, 1.0,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(frame, "Z", imagePoints[3], cv::FONT_HERSHEY_SIMPLEX, 1.0,
              cv::Scalar(255, 0, 0), 2);
}

/**
 * @brief 清空所有数据列表
 * @return void
 */
void WMIdentify::clear() {
  // 清空队列中的内容
  this->blade_tip_list.clear();
  this->wing_idx.clear();
  this->R_center_list.clear();
  this->R_idx.clear();
  this->time_list.clear();
  this->angle_list.clear();

  this->angle_velocity_list.clear();
  this->angle_velocity_list.emplace_back(
      0); // 先填充一个0，方便之后UpdateList中的数据对齐
  // 输出日志，清空成功
  // LOG_IF(INFO, this->switch_INFO) << "clear Successful";
}

/**
 * @brief 任意点位能量机关识别,角度收集和预测所需参数计算
 * @param[in] input_img     输入图像
 * @param[in] ts            串口数据
 * @return void
 */
void WMIdentify::identifyWM(cv::Mat &input_img, Translator &ts) {
  // 输入图片
  this->receive_pic(input_img);
  // 调用网络识别扇叶，结果存入fans中

  WMBlade blade;
  DetectionResult result;
  result = detect(this->img, true, false, blade);
  // cv::imshow("Processed Image", result.processedImage);
  cv::waitKey(1);

  if (blade.apex.size() == 6) {

    // 如果当前为识别到有效扇叶的第一帧（未PnP，检测到五个关键点）,进行PnP解算
    if (!pnp_solved) {
      // 构建世界坐标系中的点
      world_points.clear();
      // world_points.emplace_back(0, 0, 0.04); // R点作为原点

      // R->P2为x轴正向
      // world_points.emplace_back(this->gp->d_Radius, 0, 0);
      world_points.emplace_back(this->gp->d_RP2, 0, 0); // P2点
      world_points.emplace_back(this->gp->d_Radius, this->gp->d_P1P3 / 2,
                                0); // P1点
      world_points.emplace_back(this->gp->d_Radius, -this->gp->d_P1P3 / 2,
                                0); // P3点
      world_points.emplace_back(this->gp->d_Radius / 2, 0, 0);
      // 图像坐标系中的点
      image_points.clear();
      // image_points.push_back(blade.apex[0]);
      // image_points.push_back(blade.apex[1]);
      image_points.push_back(blade.apex[2]);
      image_points.push_back(blade.apex[3]);
      image_points.push_back(blade.apex[4]);
      image_points.push_back(blade.apex[5]);
      // 进行PnP解算获取旋转矩阵
      cv::solvePnP(world_points, image_points, camera_matrix, dist_coeffs, rvec,
                   tvec);

      cv::Rodrigues(rvec, rotation_matrix);
      // 标记为已解算，后续不再重复PnP解算（固定世界坐标系，收集角度）
      pnp_solved = true;
    }
    // 如果已经解算过（有固定世界系了）
    // 那么收集角度
    if (pnp_solved) {
       image_points.clear();
      // image_points.push_back(blade.apex[0]);
      // image_points.push_back(blade.apex[1]);
      image_points.push_back(blade.apex[2]);
      image_points.push_back(blade.apex[3]);
      image_points.push_back(blade.apex[4]);
      image_points.push_back(blade.apex[5]);
      // 进行PnP解算获取旋转矩阵
      cv::solvePnP(world_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);

      cv::Rodrigues(rvec, rotation_matrix);
      // cv::Mat R_init = cv::Mat::eye(3, 3, CV_64F);   // 初始化为单位矩阵
      // cv::Mat t_init = cv::Mat::zeros(3, 1, CV_64F); // 初始化为零向量
      // cv::Rodrigues(rvec, R_init);
      // t_init = tvec.clone();
      this->calculateAngle(blade.apex[1], rotation_matrix, tvec);
      cv::circle(result.processedImage, blade.apex[1], 5, cv::Scalar(0, 0, 255), -1);
      // cv::imshow("Processed Image", result.processedImage);
      if (angle_velocity_list.size() >= gp->list_size - 2) {
        // 如果角度收集达到阈值，那么需要开始预测了，要给Predict发alpha和phi角度，以及距离s
        // 以扇叶方向为x轴建立随动世界系，做PnP算两个角度和距离

        // 图像坐标系中的点（世界系下的点是相对固定的，所以不必重新构建world_points了）
        image_points.clear();
        image_points.push_back(blade.apex[0]);
        image_points.push_back(blade.apex[1]);
        image_points.push_back(blade.apex[2]);
        image_points.push_back(blade.apex[3]);
        image_points.push_back(blade.apex[4]);

        // 进行PnP解算获取旋转矩阵
        cv::solvePnP(world_points, image_points, camera_matrix, dist_coeffs,
                     rvec_for_predict, tvec_for_predict);

        cv::Rodrigues(rvec_for_predict, rotation_matrix_for_predict);

        this->phi =
            this->calculatePhi(rotation_matrix_for_predict, tvec_for_predict);
        this->alpha =
            this->calculateAlpha(rotation_matrix_for_predict, tvec_for_predict);
        this->s = sqrt(cv::norm(tvec_for_predict) * cv::norm(tvec_for_predict) -
                       this->gp->H_0 * this->gp->H_0);
        // this->updateList((double)ts.message.predict_time / 1000);
      } else {

        // 如果角度收集未达到阈值，则不执行updateList
        // LOG_IF(ERROR, this->switch_ERROR)
        //     << "didn't execute updateList, lack data";
      }
    } else {
      // 如果未解算过PnP，则不执行updateList
      // LOG_IF(ERROR, this->switch_ERROR)
      //     << "The PnP coordinate system has not been established yet, "
      //        "skipping this frame.";
    }
  }
}

/**
 * @brief 更新R中心点列表（弃用）
 * @return bool true:成功，false:失败
 */
bool WMIdentify::update_R_list() {
  this->preprocess();
  this->getContours();

  // 通过图像特征寻找R
  this->getValidR();
  this->selectR();

  // 更新R中心点列表
  if (this->R_center_list.size() >= 2) {
    this->R_center_list.pop_front();
  }

  if (this->R_idx.size() > 0 && this->R_contours.size() > 0) {
    this->R_center_list.emplace_back(
        cv::minAreaRect(this->R_contours[this->R_idx[0]]).center);
  } else {
    // LOG_IF(ERROR, this->switch_ERROR)
    //     << "failed to emplace_back R_center_list, lack data";
    return -1; // return 0;
  }
  return 0;
}
/**
 * @brief 角度解算和收集函数（像素点反投影回世界系）
 * @param[in] blade_tip     扇叶顶点
 * @param[in] rotation_matrix     旋转矩阵
 * @param[in] tvec            平移向量
 * @return void
 */
void WMIdentify::calculateAngle(cv::Point2f blade_tip, cv::Mat rotation_matrix,
                                cv::Mat tvec) {
  // 计算相机光心在世界坐标系中的位置
  cv::Mat camera_in_world = -rotation_matrix.t() * tvec;
  // 计算图像点在相机坐标系和世界坐标系中的向量
  double u = blade_tip.x;
  double v = blade_tip.y;

  cv::Mat direction_camera =
      (cv::Mat_<double>(3, 1) << (u - camera_matrix.at<double>(0, 2)) /
                                     camera_matrix.at<double>(0, 0),
       (v - camera_matrix.at<double>(1, 2)) / camera_matrix.at<double>(1, 1),
       1.0);

  cv::Mat direction_world = rotation_matrix.t() * direction_camera;

  // 计算射线与z=0平面的交点（反投影回世界系）
  double s =
      -camera_in_world.at<double>(2, 0) / direction_world.at<double>(2, 0);
  double X =
      camera_in_world.at<double>(0, 0) + s * direction_world.at<double>(0, 0);
  double Y =
      camera_in_world.at<double>(1, 0) + s * direction_world.at<double>(1, 0);

  this->angle = atan2(Y, X);
  std::cout << "angle: " << this->angle << std::endl;
  std::cout << "length: " << sqrt(X * X + Y * Y) << std::endl;
  std::cout << "X: " << X << std::endl;
  std::cout << "Y: " << Y << std::endl;
}
/**
 * @brief 计算phi
 * @param[in] rotation_matrix     旋转矩阵
 * @param[in] tvec            平移向量
 * @return void
 */
double WMIdentify::calculatePhi(cv::Mat rotation_matrix, cv::Mat tvec) {
  cv::Mat camera_in_world = -rotation_matrix.t() * tvec;
  // 相机z轴在世界坐标系中的向量
  cv::Mat Z_camera_in_world = rotation_matrix.t().col(2);
  double Vx = camera_in_world.at<double>(0, 0);
  double Vz = camera_in_world.at<double>(2, 0);
  double Zx = Z_camera_in_world.at<double>(0, 0);
  double Zz = Z_camera_in_world.at<double>(2, 0);
  // phi的值为二者点积除以二者模的乘积
  double phi = acos(
      Vx * Zx + Vz * Zz / (sqrt(Vx * Vx + Vz * Vz) * sqrt(Zx * Zx + Zz * Zz)));
  return phi;
}
/**
 * @brief 计算alpha
 * @param[in] rotation_matrix     旋转矩阵
 * @param[in] tvec            平移向量
 * @return void
 */
double WMIdentify::calculateAlpha(cv::Mat rotation_matrix, cv::Mat tvec) {
  cv::Mat camera_in_world = -rotation_matrix.t() * tvec;
  double Vx = camera_in_world.at<double>(0, 0);
  double Vz = camera_in_world.at<double>(2, 0);
  // phi的值为二者点积除以二者模的乘积
  double alpha = atan2(Vx, Vz);
  return alpha;
}
/**
 * @brief 接收输入图像
 * @param[in] input_img     输入图像
 * @return void
 */
void WMIdentify::receive_pic(cv::Mat &input_img) {
  this->img_0 = input_img.clone();
  this->img = input_img.clone();
  // LOG_IF(INFO, this->switch_INFO) << "receive_pic Successful";
}

/**
 * @brief 预处理图像
 * @return void
 */
void WMIdentify::preprocess() {

#ifdef DEBUGHIT
  this->img_0 = this->img.clone();

#endif // DEBUGHIT
  // 将蒙板中需要取到的区域（在globalParam中调整）内的像素点变为白色

#ifdef DEBUGMODE
  // NOTE: 将来优化代码可以减少克隆次数
  cv::Mat mask(img.rows, img.cols, CV_8UC1,
               cv::Scalar(0)); //!< 创建的单通道图像，用于蒙板
  cv::Mat dst1;
  // 展示蒙板之后的结果
  mask(cv::Rect(this->img.cols * this->gp->mask_TL_x,
                this->img.rows * this->gp->mask_TL_y,
                this->img.cols * this->gp->mask_width,
                this->img.rows * this->gp->mask_height)) = 255;
  // 实现蒙板，将两边与能量机关无关的部分去掉
  this->img.copyTo(dst1, mask); //!< 于储存蒙板之后的图片
  cv::imshow("after mask", dst1);
  cv::imwrite("./imgs/after_mask.jpg", dst1);
  if (this->gp->switch_gaussian_blur == ON) {
    cv::GaussianBlur(dst1, dst1, cv::Size(5, 5), 0.0, 0.0);
    cv::imwrite("./imgs/after_mask.jpg", dst1);
  }
  // hsv二值化，此部分的参数对于后续的操作尤为重要

  cv::cvtColor(dst1, dst1, cv::COLOR_BGR2HSV);
  cv::imwrite("./imgs/hsv.jpg", dst1);
  // cv::inRange(dst1, cv::Scalar(this->gp->hmin, this->gp->smin,
  // this->gp->vmin), cv::Scalar(this->gp->hmax, this->gp->smax,
  // this->gp->vmax), this->binary);
  cv::Mat lowerRedMask, upperRedMask, redMask;
  cv::inRange(dst1, cv::Scalar(0, 100, 50), cv::Scalar(20, 255, 255),
              lowerRedMask); // 低范围
  cv::inRange(dst1, cv::Scalar(170, 100, 50), cv::Scalar(180, 255, 255),
              upperRedMask); // 高范围
  // 合并两个掩码
  cv::bitwise_or(lowerRedMask, upperRedMask, this->binary);

  // 提取红色部分
  cv::Mat redParts;
  // cv::bitwise_and(this->img, this->img, redParts, redMask);
  // cv::imwrite("./imgs/red_part.jpg", redMask);
  // DEBUGMODE
#else
  if (this->gp->switch_gaussian_blur == ON) {
    cv::GaussianBlur(this->img, this->img, cv::Size(5, 5), 0.0, 0.0);
  }
  cv::cvtColor(this->img, this->img, cv::COLOR_BGR2HSV);
  cv::inRange(
      this->img, cv::Scalar(this->gp->hmin, this->gp->smin, this->gp->vmin),
      cv::Scalar(this->gp->hmax, this->gp->smax, this->gp->vmax), this->binary);

#endif
#ifdef DEBUGMODE
  // 展示预处理之后的binary
  cv::imshow("binary after preprocess", this->binary);
  cv::imwrite("./imgs/binary_after_preprocess.jpg", this->binary);
#endif // DEBUGMODE

  // 日志输出预处理成功
  // LOG_IF(INFO, this->switch_INFO == ON) << "preprocess successful";
}

/**
 * @brief 获取图像中的轮廓
 * @return void
 */
void WMIdentify::getContours() {
  // 清空轮廓信息
  this->R_contours.clear();
  this->wing_contours.clear();
  cv::Mat binary_clone = this->binary;
  // 进行图形学操作，最后新添加的膨胀后腐蚀可以让灯带连在一起
  cv::dilate(
      binary_clone, binary_clone,
      cv::getStructuringElement(
          cv::MORPH_RECT, cv::Size(this->gp->dialte1, this->gp->dialte1)));
  // 检测最外围轮廓，储存找到的轮廓至R_contours，用于R的识别
  cv::findContours(binary_clone, this->R_contours, CV_RETR_EXTERNAL,
                   CV_CHAIN_APPROX_NONE);

#ifdef DEBUGMODE
  cv::imshow("R", binary_clone);
  cv::imwrite("./imgs/R.jpg", binary_clone);
#endif // DEBUGMODE
#ifndef USEWMNET
  // 进行获取装甲板需要的图形学操作
  cv::dilate(
      binary_clone, binary_clone,
      cv::getStructuringElement(
          cv::MORPH_RECT, cv::Size(this->gp->dialte2, this->gp->dialte2)));
  cv::dilate(
      binary_clone, binary_clone,
      cv::getStructuringElement(
          cv::MORPH_RECT, cv::Size(this->gp->dialte3, this->gp->dialte3)));
  // 提取binary中所有轮廓，且hierarchy是以“树状结构”来进行组织，储存找到的轮廓至wing_contours，用于装甲板的识别
  cv::findContours(binary_clone, this->wing_contours, this->hierarchy,
                   CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
#endif
#ifdef DEBUGMODE
  cv::imshow("armor", binary_clone);
  cv::imshow("./imgs/armor.jpg", binary_clone);
#endif // DEBUGMODE
  // 日志输出获得轮廓成功成功
  // LOG_IF(INFO, this->switch_INFO == ON) << "getContours successful";
}

/**
 * @brief 获取有效的R点
 * @return int R点的状态(0:无R点, 1:一个R点, 2:多个R点)
 */
int WMIdentify::getValidR() {
  this->R_idx.clear();
  // 遍历全部的轮廓，通过轮廓特征判断是否是R
  for (int i = 0; i < this->R_contours.size(); i++) {
    if (IsValidR(this->R_contours[i], *gp)) {
      this->R_idx.emplace_back(i);
    }
  }
  // 通过当前找到的符合的R的数量，输出当前状态
  if (this->R_idx.size() == 0) {
    // LOG_IF(INFO, this->switch_INFO) << "getValidR Get No R";
    this->R_stat = 0;
  } else if (this->R_idx.size() == 1) {
    // LOG_IF(INFO, this->switch_INFO) << "getValidR Get One R";
    this->R_stat = 1;
  } else if (this->R_idx.size() > 1) {
    // LOG_IF(INFO, this->switch_INFO)
    //     << "getValidR Get Many R:" << this->R_idx.size();
    this->R_stat = 2;
  }
  return this->R_stat;
}

/**
 * @brief 从多个R点中选择最合适的一个
 * @return int 选择状态(0:失败, 1:成功)
 */
int WMIdentify::selectR() {
  // 假如没有R，识别失败
  if (this->R_stat == 0) {
    // LOG_IF(INFO, this->switch_INFO == ON) << "selectR failed";
    return 0;
  }
  // 假如只有一个R，没有选择的必要，直接成功
  else if (this->R_stat == 1) {
    // LOG_IF(INFO, this->switch_INFO == ON) << "selectR successful";
    // 如果UI开关打开
    if (this->gp->switch_UI == ON) {
      // 遍历idx里面每一个索引(其实只有一个，保证结构一致性，不更改)
      for (int i = 0; i < R_idx.size(); i++) {
        // 如果轮廓显示开关打开，画出当前找到的R的轮廓
        if (this->gp->switch_UI_contours == ON)
          cv::drawContours(this->img, this->R_contours, R_idx[i],
                           cv::Scalar(0, 0, 255), 3);
        // 如果面积显示开关打开，显示出当前找到的R的面积、紧致度、圆度
        if (this->gp->switch_UI_areas == ON) {
          double s_area = cv::contourArea(R_contours[R_idx[i]]);
          cv::putText(this->img, std::to_string(s_area),
                      cv::minAreaRect(this->R_contours[this->R_idx[i]]).center,
                      1, 1, cv::Scalar(255, 255, 0));
          double girth = cv::arcLength(R_contours[R_idx[i]], true);
          double compactness = (girth * girth) / s_area;
          double circularity = (4 * Pi * s_area) / (girth * girth);
          cv::putText(this->img, std::to_string(compactness),
                      cv::minAreaRect(this->R_contours[this->R_idx[i]]).center +
                          cv::Point2f(0, 20),
                      1, 1, cv::Scalar(255, 255, 0));
          cv::putText(this->img, std::to_string(circularity),
                      cv::minAreaRect(this->R_contours[this->R_idx[i]]).center +
                          cv::Point2f(0, 40),
                      1, 1, cv::Scalar(255, 255, 0));
          cv::imwrite("./imgs/R_SHOW.jpg", this->img);
        }
      }
    }
    return 1;
  }
  // 假如有很多R，开始筛选
  else if (this->R_stat == 2) {
    // 如果UI开关打开
    if (this->gp->switch_UI == ON) {
      // 遍历idx里面每一索引
      for (int i = 0; i < R_idx.size(); i++) {
        // 如果轮廓显示开关打开，画出当前找到的R的轮廓
        if (this->gp->switch_UI_contours == ON)
          cv::drawContours(this->img, this->R_contours, R_idx[i],
                           cv::Scalar(0, 0, 255), 3);
        // 如果面积显示开关打开，显示出当前找到的R的面积、紧致度、圆度
        if (this->gp->switch_UI_areas == ON) {
          double s_area = cv::contourArea(R_contours[R_idx[i]]);
          cv::putText(this->img, std::to_string(s_area),
                      cv::minAreaRect(this->R_contours[this->R_idx[i]]).center,
                      1, 1, cv::Scalar(255, 255, 0));
          double girth = cv::arcLength(R_contours[R_idx[i]], true);
          double compactness = (girth * girth) / s_area;
          double circularity = (4 * Pi * s_area) / (girth * girth);
          cv::putText(this->img, std::to_string(compactness),
                      cv::minAreaRect(this->R_contours[this->R_idx[i]]).center +
                          cv::Point2f(0, 20),
                      1, 1, cv::Scalar(255, 255, 0));
          cv::putText(this->img, std::to_string(circularity),
                      cv::minAreaRect(this->R_contours[this->R_idx[i]]).center +
                          cv::Point2f(0, 40),
                      1, 1, cv::Scalar(255, 255, 0));
        }
      }
    }
    // 遍历全部可能是R的索引，判断是否在中心区域，假如是，认为它是R，让R_idx首位是它，结束筛选
    // for (int i = 0; i < this->R_idx.size(); i++)
    // {
    //     if (cv::minAreaRect(this->R_contours[this->R_idx[i]]).center.x >
    //     this->img.cols * this->gp->R_roi_xl &&
    //     cv::minAreaRect(R_contours[R_idx[i]]).center.x < this->img.cols *
    //     this->gp->R_roi_xr &&
    //     cv::minAreaRect(R_contours[R_idx[i]]).center.y > this->img.rows *
    //     this->gp->R_roi_yb &&
    //     cv::minAreaRect(R_contours[R_idx[i]]).center.y < this->img.rows *
    //     this->gp->R_roi_yt)
    //     {
    //         this->R_idx[0] = this->R_idx[i];
    //         LOG_IF(INFO, this->switch_INFO == ON) << "selectR successful";
    //     }
    // }
    std::sort(R_idx.begin(), R_idx.end(), [this](int &a, int &b) {
      return calculateDistanceSquare(cv::minAreaRect(R_contours[a]).center,
                                     R_estimate) <
             calculateDistanceSquare(cv::minAreaRect(R_contours[b]).center,
                                     R_estimate);
    });
    if (this->R_idx.size() == 1) {
      // LOG_IF(INFO, this->switch_INFO == ON) << "selectR successful";
      return 1;
    } else if (this->R_idx.size() == 0) {
      // LOG_IF(INFO, this->switch_INFO == ON) << "selectR failed, no R left";
      return 0;
    } else if (this->R_idx.size() > 1) {
      // LOG_IF(INFO, this->switch_INFO == ON)
      //     << "selectR failed, too many R left";
      return 0;
    }
  }
  // LOG_IF(INFO, this->switch_INFO == ON) << "selectR failed";
  return 0;
}

/**
 * @brief 判断是否是R
 * @param[in] contour 轮廓
 * @param[in] gp 全局变量结构体
 * @return bool 是否是R
 */
bool WMIdentify::IsValidR(std::vector<cv::Point> contour, GlobalParam &gp) {
  double Pi = 3.1415926;
  double s_area = cv::contourArea(contour);
  double girth = cv::arcLength(contour, true);
  // if (s_area < 150)
  //     return false;
  // if (s_area < gp.s_R_min || s_area > gp.s_R_max)
  // {
  //     LOG_IF(INFO, gp.switch_INFO) << "IsValidR Successful But Not R,s_area
  //     wrong:" << s_area; return false;
  // }
  if (s_area < 0 || s_area > gp.s_R_max) {
    // LOG_IF(INFO, gp.switch_INFO)
    //     << "IsValidR Successful But Not R,s_area wrong:" << s_area;
    return false;
  }
  cv::RotatedRect R_rect = minAreaRect(contour);
  cv::Size2f R_size = R_rect.size;
  float length = R_size.height > R_size.width
                     ? R_size.height
                     : R_size.width; // 将矩形的长边设置为长
  float width = R_size.height < R_size.width
                    ? R_size.height
                    : R_size.width; // 将矩形的短边设置为宽
  float lw_ratio = length / width;
  float s_ratio = s_area / R_size.area();
  if (lw_ratio < gp.R_ratio_min || lw_ratio > gp.R_ratio_max) {
    // LOG_IF(INFO, gp.switch_INFO)
    //     << "IsValidR Successful But Not R,lw_ratio wrong:" << lw_ratio
    //     << " and s_area is:" << s_area;
    return false;
  }
  if (s_ratio < gp.s_R_ratio_min || s_ratio > gp.s_R_ratio_max) {
    // LOG_IF(INFO, gp.switch_INFO)
    //     << "IsValidR Successful But Not R,s_ratio wrong:" << s_ratio
    //     << " and s_area is:" << s_area;
    return false;
  }
  double compactness = (girth * girth) / s_area;
  double circularity = (4 * Pi * s_area) / (girth * girth);
  if (circularity < gp.R_circularity_min ||
      circularity > gp.R_circularity_max) {
    // LOG_IF(INFO, gp.switch_INFO)
    //     << "IsValidR Successful But Not R,circularity wrong:" << circularity
    //     << " and s_area is:" << s_area;
    return false;
  }
  if (compactness < gp.R_compactness_min ||
      compactness > gp.R_compactness_max) {
    // LOG_IF(INFO, gp.switch_INFO)
    //     << "IsValidR Successful But Not R,compactness wrong:" << compactness
    //     << " and s_area is:" << s_area;
    return false;
  }
  // LOG_IF(INFO, gp.switch_INFO)
  //     << "IsValidR Successful,compactness:" << compactness
  //     << " circularity:" << circularity << " s_area:" << s_area;
  return true;
}

/**
 * @brief 更新数据列表
 * @param[in] time     当前时间
 * @return void
 */
void WMIdentify::updateList(double time) {

#ifdef DEBUGMODE
  if (this->R_center_list.size() > 0 && ready_to_update)
    cv::circle(this->img, this->R_center_list[this->R_center_list.size() - 1],
               10, cv::Scalar(0, 255, 255), -1);
  if (this->blade_tip_list.size() > 0 && ready_to_update)
    cv::circle(this->img, this->blade_tip_list[this->blade_tip_list.size() - 1],
               10, cv::Scalar(0, 0, 255), -1);
#endif
  // 更新时间队列
  if (this->time_list.size() >= this->gp->list_size &&
      gp->gap % gp->gap_control == 0) {
    this->time_list.pop_front();
  }
  if (ready_to_update && gp->gap % gp->gap_control == 0) {
    this->time_list.push_back(time);
  }
  // 更新角度队列
  if (this->angle_list.size() >= this->gp->list_size &&
      gp->gap % gp->gap_control == 0) {
    this->angle_list.pop_front();
  }
  if (ready_to_update && gp->gap % gp->gap_control == 0) {
    this->angle_list.push_back(angle);
  }
  // 更新phi队列
  if (this->phi_list.size() >= this->gp->list_size) {
    this->phi_list.pop_front();
  }
  if (ready_to_update) {
    this->phi_list.push_back(phi);
  }
  // 更新alpha队列
  if (this->alpha_list.size() >= this->gp->list_size) {
    this->alpha_list.pop_front();
  }
  if (ready_to_update) {
    this->alpha_list.push_back(alpha);
  }

  // 更新角速度队列
  if (this->angle_velocity_list.size() >= this->gp->list_size &&
      gp->gap % gp->gap_control == 0) {
    this->angle_velocity_list.pop_front();
  }
  if (angle_list.size() > 1 && time_list.size() > 1 && ready_to_update &&
      gp->gap % gp->gap_control == 0) {
    // 计算角度变化量
    double dangle = this->angle_list[this->angle_list.size() - 1] -
                    this->angle_list[this->angle_list.size() - 2];
    // 防止数据跳变
    dangle += (abs(dangle) > Pi) ? 2 * Pi * (-dangle / abs(dangle)) : 0;
    // 计算时间变化量
    double dtime = (this->time_list[this->time_list.size() - 1] -
                    this->time_list[this->time_list.size() - 2]);
    // 更新角速度队列,简单检验一下数据，获得扇叶切换时间
    if (abs(dangle / dtime) > 5) {
      this->FanChangeTime = time_list.back() * 1000;
      this->time_list.pop_front();
      this->angle_list.pop_front();
      gp->gap--;
    } else {
      this->angle_velocity_list.emplace_back(dangle / dtime);
    }

    // 更新旋转方向
    // std::cout<<this->time_list.back()<<std::endl;
    // std::cout<<" "<<this->angle_velocity_list.back()<<std::endl;
    this->direction = 0;
    for (int i = 0; i < angle_velocity_list.size(); i++) {
      this->direction += this->angle_velocity_list[i];
    }
  }
  // 更新gap
  if (this->ready_to_update) {
    gp->gap++;
  }
  // 输出日志，更新队列成功
  // LOG_IF(INFO, this->switch_INFO == ON) << "updateList successful";
}

/**
 * @brief 获取时间列表
 * @return std::deque<double> 时间列表
 */
std::deque<double> WMIdentify::getTimeList() { return this->time_list; }

/**
 * @brief 获取角速度列表
 * @return std::deque<double> 角速度列表
 */
std::deque<double> WMIdentify::getAngleVelocityList() {
  return this->angle_velocity_list;
}

/**
 * @brief 获取旋转方向
 * @return double 旋转方向
 */
double WMIdentify::getDirection() { return this->direction; }

/**
 * @brief 获取最新角度
 * @return double 最新角度值
 */
double WMIdentify::getAngleList() {
  return this->angle_list[angle_list.size() - 1];
}

/**
 * @brief 获取R点中心坐标
 * @return cv::Point2d R点中心坐标
 */
cv::Point2d WMIdentify::getR_center() {
  return this->R_center_list[R_center_list.size() - 1];
}

/**
 * @brief 获取半径
 * @return double 半径值
 */
double WMIdentify::getRadius() {
  return sqrt(
      calculateDistanceSquare(this->R_center_list[R_center_list.size() - 1],
                              this->blade_tip_list[blade_tip_list.size() - 1]));
}

double WMIdentify::getPhi() { return this->phi; }

double WMIdentify::getAlpha() { return this->alpha; }

/**
 * @brief 获取列表状态
 * @return int 列表状态
 */
int WMIdentify::getListStat() { return this->list_stat; }

/**
 * @brief 获取原始图像
 * @return cv::Mat 原始图像
 */
cv::Mat WMIdentify::getImg0() { return this->img_0; }

/**
 * @brief 获取数据图像
 * @return cv::Mat 数据图像
 */
cv::Mat WMIdentify::getData_img() { return this->data_img; }

/**
 * @brief 清空速度相关数据
 * @return void
 */
void WMIdentify::ClearSpeed() {
  this->angle_velocity_list.clear();
  this->time_list.clear();
}

/**
 * @brief 获取扇叶切换时间
 * @return uint32_t 扇叶切换时间
 */
uint32_t WMIdentify::GetFanChangeTime() { return this->FanChangeTime; }

/**
 * @brief 根据翻译器状态判断是否需要清空数据
 * @param[in] translator     翻译器对象
 * @return void
 */
void WMIdentify::JudgeClear(Translator translator) {
  if (translator.message.status % 5 == 0) // 进入自瞄便清空识别的所有数据
    this->clear();
}

/**
 * @brief 计算两点之间的距离平方
 * @param[in] p1 点1
 * @param[in] p2 点2
 * @return double 距离平方
 */
double WMIdentify::calculateDistanceSquare(cv::Point2f p1, cv::Point2f p2) {
  return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}
