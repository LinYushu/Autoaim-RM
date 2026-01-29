#include "data_manager/video_source.h"
#include <iostream>
#include <chrono>

VideoSource::VideoSource(const std::string& source_path, int width, int height) 
    : source_path_(source_path) {
    
    // 1. 初始化父类成员
    this->width = width;
    this->height = height;
    this->camera_id = 1;

    // [修复] 正确初始化 SwapBuffer (无参数)
    this->buffer = new rm::SwapBuffer<rm::Frame>();

    // 2. 打开视频源
    if (source_path == "0") {
        cap_.open(0); // 打开笔记本自带摄像头
    } else {
        cap_.open(source_path); // 打开视频文件
    }

    if (!cap_.isOpened()) {
        std::cerr << "[VideoSource] Error: Cannot open source " << source_path << std::endl;
    } else {
        std::cout << "[VideoSource] Successfully opened " << source_path << std::endl;
    }

    running_ = false;
}

VideoSource::~VideoSource() {
    running_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
    if (cap_.isOpened()) {
        cap_.release();
    }
    // 注意：this->buffer 会由父类 Camera 的析构函数自动 delete，这里不需要手动 delete
}

void VideoSource::startCapture() {
    if (running_) return;
    running_ = true;
    thread_ = std::thread(&VideoSource::captureThread, this);
}

void VideoSource::captureThread() {
    cv::Mat frame_img;
    // 计算每帧间隔的毫秒数 (30fps -> 33ms)
    int delay_ms = 1000 / target_fps_;

    while (running_) {
        auto start_time = std::chrono::steady_clock::now();

        if (!cap_.isOpened()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }

        cap_ >> frame_img;
        
        if (frame_img.empty()) {
            // 如果是视频文件，播放完后重头开始
            if (source_path_ != "0") {
                cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            }
            continue;
        }

        // 调整尺寸以匹配训练模型的要求
        cv::Mat resized_img;
        cv::resize(frame_img, resized_img, cv::Size(this->width, this->height));

        // 构造 Frame
        std::shared_ptr<rm::Frame> frame = std::make_shared<rm::Frame>();
        
        // [修复] 赋值 image
        frame->image = std::make_shared<cv::Mat>(resized_img.clone());
        
        // [修复] 使用 time_point 和全局 getTime()
        frame->time_point = getTime(); 
        
        frame->width = this->width;
        frame->height = this->height;
        frame->camera_id = this->camera_id;

        // 推入缓冲区
        this->buffer->push(frame);

        // 控制帧率
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        if (elapsed < delay_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms - elapsed));
        }
    }
}