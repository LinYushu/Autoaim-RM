#ifndef VIDEO_SOURCE_H
#define VIDEO_SOURCE_H

#include <string>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "structure/camera.hpp" // 包含 rm::Camera, rm::SwapBuffer
#include "structure/stamp.hpp"  // 包含 rm::Frame
#include "utils/timer.h"        // 包含 getTime()

class VideoSource : public rm::Camera {
public:
    // source_path: "0" 表示使用Webcam，或者传入视频路径
    VideoSource(const std::string& source_path, int width, int height);
    ~VideoSource();

    void startCapture();

private:
    void captureThread();

    cv::VideoCapture cap_;
    std::thread thread_;
    std::atomic<bool> running_;
    std::string source_path_;
    int target_fps_ = 30; // 模拟帧率
};

#endif