#include "MessageManager.hpp"
#include "SerialPort.hpp"
#include "WMIdentify.hpp"
#include "WMPredict.hpp"
#include "camera.hpp"
#include "globalParam.hpp"
#include "globalText.hpp"
#include <AimAuto.hpp>
#include <UIManager.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <glog/logging.h>
#include <monitor.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <pthread.h>
#include <unistd.h>
#define RESIZE 1

// 全局变量参数，这个参数存储着全部的需要的参数
GlobalParam gp;
// 通信类
MessageManager MManager(gp);

// 定义双缓冲区
struct DataBuffer {
  Translator translator;
  cv::Mat pic;
  bool data_ready;   // 标志数据是否准备好
  double time_stamp; // 时间戳
};
DataBuffer buffers[2]; // 两个缓冲区

// 定义线程锁和条件变量
pthread_mutex_t Mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t Barrier;

// 定义退出标志位
bool exit_flag = false;

// 读线程，负责读取串口信息以及取流
void *ReadFunction(void *arg);
// 运算线程，负责对图片进行处理，并且完成后续的要求
void *OperationFunction(void *arg);

int main(int argc, char **argv) {
  // 初始化Glog并设置部分标志位
  // google::InitGoogleLogging(argv[0]);
  // 设置Glog输出的log文件写在address中log_address对应的地址下
  // FLAGS_log_dir = "../log";
  printf("welcome\n");
// 实例化通信串口类
printf("222\n");
SerialPort *serialPort = new SerialPort(argv[1]);
printf("333\n");
// 设置通信串口对象初始值
serialPort->InitSerialPort(int(*argv[2] - '0'), 8, 1, 'N');
printf("555\n");

  // 初始化线程锁和条件变量
  pthread_barrier_init(&Barrier, nullptr, 2);
  // 输出日志，开始初始化
  pthread_t readThread;
  pthread_t operationThread;

  // 开启线程
  pthread_create(&readThread, NULL, ReadFunction, serialPort);
  pthread_create(&operationThread, NULL, OperationFunction, serialPort);
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(0, &cpuset);

  // 等待线程结束
  pthread_join(readThread, NULL);
  pthread_join(operationThread, NULL);

  // 销毁线程锁和条件变量
  pthread_mutex_destroy(&Mutex);
  pthread_barrier_destroy(&Barrier);

  return 0;
}

void *ReadFunction(void *arg) // 读线程
{
  int current_buffer = 0; // 当前使用的缓冲区
  int cnt = 0;

  // 传入的参数赋给串口，以获得串口数据
  SerialPort *serialPort = (SerialPort *)arg;
  while (1) {
    pthread_mutex_lock(&Mutex);
    // usleep(1000);
    // printf("read thread is running %d %d\n", ++cnt, current_buffer);
    MManager.read(buffers[current_buffer].translator, *serialPort);
    if (buffers[current_buffer].translator.message.status % 5 != 0) {

      gp.attack_mode = ENERGY;
    } else {
      gp.attack_mode = ARMOR;
    }


    buffers[current_buffer].time_stamp =
        std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();

    // usleep(200 * 1000);
    pthread_mutex_unlock(&Mutex);
    // 切换缓冲区
    current_buffer = (current_buffer + 1) % 2;

    pthread_barrier_wait(&Barrier);
    // printf("read thread is end %d %d\n", cnt, current_buffer);
  }
  return NULL;
}

void *OperationFunction(void *arg) {
  SerialPort *serialPort = (SerialPort *)arg;
  // 实例化能量机关识别类
  WMIdentify WMI(gp);
  // 实例化自瞄类
  AimAuto aim(&gp);
  // 实例化UI类
  UIManager UI(gp);
  // 重置能量机关识别类
  WMI.clear();
  // 实例化能量机关预测类
  WMIPredict WMIPRE;
  cv::Mat pic;
  Translator translator;
  double dt = 0;
  double last_time_stamp = 0;


  uint8_t error_times{0};
  int processing_buffer = 1; // 当前处理的缓冲区
  int cnt = 0;

  translator.message.yaw = 30;
  MManager.write(translator, *serialPort);
  if (translator.message.status == 99)
    exit(0);

  pthread_barrier_wait(&Barrier);
    
  return NULL;
}
