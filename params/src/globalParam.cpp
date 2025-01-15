#include "globalParam.hpp"
#include <glog/logging.h>
#include <filesystem>
#include <iostream>
void GlobalParam::initGlobalParam(const int color)
{   
    cv::FileStorage fs;
    this ->color = color;

    // 打开CameraConfig配置文件
    if(!fs.open("../config/CameraConfig.yaml", cv::FileStorage::READ)){
        printf("CameraConfig.yaml not found!\n");
        exit(1);
    }
    fs["cam_index"] >> cam_index;
    fs["enable_auto_exp"] >> enable_auto_exp;
    fs["energy_exp_time"] >> energy_exp_time;
    fs["armor_exp_time"] >> armor_exp_time;
    fs["r_balance"] >> r_balance;
    fs["g_balance"] >> g_balance;
    fs["b_balance"] >> b_balance;
    fs["e_r_balance"] >> e_r_balance;
    fs["e_g_balance"] >> e_g_balance;
    fs["e_b_balance"] >> e_b_balance;
    fs["enable_auto_gain"] >> enable_auto_gain;
    fs["gain"] >> gain;
    fs["gamma_value"] >> gamma_value;
    fs["trigger_activation"] >> trigger_activation;
    fs["frame_rate"] >> frame_rate;
    fs["enable_trigger"] >> enable_trigger;
    fs["trigger_source"] >> trigger_source;
    fs["cx"] >> cx;  // cx
    fs["cy"] >> cy;  // cy
    fs["fx"] >> fx;  // fx
    fs["fy"] >> fy;  // fy
    fs["k1"] >> k1;
    fs["k2"] >> k2;
    fs["k3"] >> k3;
    fs["p1"] >> p1;
    fs["p2"] >> p2;
    fs.release();
    
    // 打开AimautoConfig配置文件
    if(!fs.open("../config/AimautoConfig.yaml", cv::FileStorage::READ)){
        printf("AimautoConfig.yaml not found!\n");
        exit(1);
    }
    fs["cost_threshold"] >> cost_threshold;
    fs["max_lost_frame"] >> max_lost_frame;
    // 卡尔曼滤波相关参数
    fs["s2qxyz"] >> s2qxyz;              // 位置转移噪声
    fs["s2qyaw"] >> s2qyaw;              // 角度转移噪声
    fs["s2qr"] >> s2qr;                  // 半径转移噪声
    fs["r_xy_factor"] >> r_xy_factor;     
    fs["r_z"] >> r_z;     
    fs["r_yaw"] >> r_yaw;
    fs["s2p0xyr"] >> s2p0xyr;
    fs["s2p0yaw"] >> s2p0yaw;
    fs["r_initial"] >> r_initial;
    fs.release();

    // 打开DetectionConfig配置文件
    if(!fs.open("../config/DetectionConfig.yaml", cv::FileStorage::READ)){
        printf("DetectionConfig.yaml not found!\n");
        exit(1);
    }
    fs["min_ratio"] >> min_ratio;
    fs["max_ratio"] >> max_ratio;
    fs["max_angle_l"] >> max_angle_l;
    fs["min_light_ratio"] >> min_light_ratio;
    fs["min_small_center_distance"] >> min_small_center_distance;
    fs["max_small_center_distance"] >> max_small_center_distance;
    fs["min_large_center_distance"] >> min_large_center_distance;
    fs["max_large_center_distance"] >> max_large_center_distance;
    fs["max_angle_a"] >> max_angle_a;
    fs["num_threshold"] >> num_threshold;
    fs["blue_threshold"] >> blue_threshold;
    fs["red_threshold"] >> red_threshold;
    fs["grad_max"] >> grad_max;
    fs["grad_min"] >> grad_min;
    fs.release();

    if(!fs.open("../config/WindmillConfig.yaml", cv::FileStorage::READ)){
        printf("WindmillConfig.yaml not found!\n");
        exit(1);
    }
    fs["mask_TL_x"] >> mask_TL_x;
    fs["mask_TL_y"] >> mask_TL_y;
    fs["mask_width"] >> mask_width;
    fs["mask_height"] >> mask_height;

    fs["hmin"] >> hmin;
    fs["hmax"] >> hmax;
    fs["smin"] >> smin;
    fs["smax"] >> smax;
    fs["vmin"] >> vmin;
    fs["vmax"] >> vmax;

    fs["e_hmin"] >> e_hmin;
    fs["e_hmax"] >> e_hmax;
    fs["e_smin"] >> e_smin;
    fs["e_smax"] >> e_smax;
    fs["e_vmin"] >> e_vmin;
    fs["e_vmax"] >> e_vmax;

    fs["switch_gaussian_blur"] >> switch_gaussian_blur;

    fs["switch_UI_contours"] >> switch_UI_contours;
    fs["switch_UI_areas"] >> switch_UI_areas;
    fs["switch_UI"] >> switch_UI;

    fs["s_R_min"] >> s_R_min;
    fs["s_R_max"] >> s_R_max;
    fs["R_ratio_min"] >> R_ratio_min;
    fs["R_ratio_max"] >> R_ratio_max;
    fs["s_R_ratio_min"] >> s_R_ratio_min;
    fs["s_R_ratio_max"] >> s_R_ratio_max;
    fs["R_circularity_min"] >> R_circularity_min;
    fs["R_circularity_max"] >> R_circularity_max;
    fs["R_compactness_min"] >> R_compactness_min;
    fs["R_compactness_max"] >> R_compactness_max;
    fs["R"] >> R;
    fs["length"] >> length;
    fs["H_0"] >> H_0;
    fs["hit_dx"] >> hit_dx;
    fs["constant_speed"] >> constant_speed;
    fs["direction"] >> direction;
    fs["init_k_"] >> init_k_;
    fs["d_Radius"] >> d_Radius;
    fs["d_RP2"] >> d_RP2;
    fs["d_P1P3"] >> d_P1P3;

    fs["A"] >> A;
    fs["w"] >> w;
    fs["fai"] >> fai;

    fs["dialte1"] >> dialte1;
    fs["dialte2"] >> dialte2;
    fs["dialte3"] >> dialte3;

    // fs["gap"] >> gap;
    // fs["gap_control"] >> gap_control;
    // fs["min_bullet_v"] >> min_bullet_v;

    // fs["WM_cx"] >> WM_cx;
    // fs["WM_cy"] >> WM_cy;
    // fs["WM_fx"] >> WM_fx;
    // fs["WM_fy"] >> WM_fy;

    // fs["WM_k1"] >> WM_k1;
    // fs["WM_k2"] >> WM_k2;
    // fs["WM_k3"] >> WM_k3;
    // fs["WM_p1"] >> WM_p1;
    // fs["WM_p2"] >> WM_p2;

    fs["R_roi_xl"] >> R_roi_xl;
    fs["R_roi_yt"] >> R_roi_yt;
    fs["R_roi_xr"] >> R_roi_xr;
    fs["R_roi_yb"] >> R_roi_yb;

    fs["list_size"] >> list_size;
    fs.release();




    
    // LOG_IF(INFO, switch_INFO) << "initGlobalParam Successful";
}

void GlobalParam::saveGlobalParam()
{
    cv::FileStorage fs;

    // 打开CameraConfig配置文件以写入参数
    if(!fs.open("../config/CameraConfig.yaml", cv::FileStorage::WRITE)){
        printf("CameraConfig.yaml not found!\n");
        exit(1);
    }
    fs << "cam_index" << cam_index;
    fs << "enable_auto_exp" << enable_auto_exp;
    fs << "energy_exp_time" << energy_exp_time;
    fs << "armor_exp_time" << armor_exp_time;
    fs << "r_balance" << r_balance;
    fs << "g_balance" << g_balance;
    fs << "b_balance" << b_balance;
    fs << "enable_auto_gain" << enable_auto_gain;
    fs << "gain" << gain;
    fs << "gamma_value" << gamma_value;
    fs << "trigger_activation" << trigger_activation;
    fs << "frame_rate" << frame_rate;
    fs << "enable_trigger" << enable_trigger;
    fs << "trigger_source" << trigger_source;
    fs << "cx" << cx;  // cx
    fs << "cy" << cy;  // cy
    fs << "fx" << fx;  // fx
    fs << "fy" << fy;  // fy
    fs << "k1" << k1;
    fs << "k2" << k2;
    fs << "k3" << k3;
    fs << "p1" << p1;
    fs << "p2" << p2;
    fs.release();

    // 打开AimautoConfig配置文件以写入参数
    if(!fs.open("../config/AimautoConfig.yaml", cv::FileStorage::WRITE)){
        printf("AimautoConfig.yaml not found!\n");
        exit(1);
    }
    fs << "cost_threshold" << cost_threshold;
    fs << "max_lost_frame" << max_lost_frame;
    // 卡尔曼滤波相关参数
    fs << "s2qxyz" << s2qxyz;                // 位置转移噪声
    fs << "s2qyaw" << s2qyaw;                // 角度转移噪声
    fs << "s2qr" << s2qr;                    // 半径转移噪声
    fs << "r_xy_factor" << r_xy_factor;     
    fs << "r_z" << r_z;    
    fs << "r_yaw" << r_yaw;    
    fs << "s2p0xyr" << s2p0xyr;
    fs << "s2p0yaw" << s2p0yaw;
    fs << "r_initial" << r_initial;
    fs.release();              

    // 打开DetectionConfig配置文件以写入参数
    if(!fs.open("../config/DetectionConfig.yaml", cv::FileStorage::WRITE)){
        printf("DetectionConfig.yaml not found!\n");
        exit(1);
    }
    fs << "min_ratio" << min_ratio;
    fs << "max_ratio" << max_ratio;
    fs << "max_angle_l" << max_angle_l;
    fs << "min_light_ratio" << min_light_ratio;
    fs << "min_small_center_distance" << min_small_center_distance;
    fs << "max_small_center_distance" << max_small_center_distance;
    fs << "min_large_center_distance" << min_large_center_distance;
    fs << "max_large_center_distance" << max_large_center_distance;
    fs << "max_angle_a" << max_angle_a;
    fs << "num_threshold" << num_threshold;
    fs << "blue_threshold" << blue_threshold;
    fs << "red_threshold" << red_threshold;
    fs << "grad_max" << grad_max;
    fs << "grad_min" << grad_min;
    fs.release();

    // LOG_IF(INFO, switch_INFO) << "saveGlobalParam Successful";
}