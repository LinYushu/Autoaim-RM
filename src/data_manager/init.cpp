#include <string>
#include <vector>
#include <thread>
#include <iostream>
#include <fstream>
#include <chrono>
#include "data_manager/base.h"
#include "data_manager/param.h"
#include "data_manager/video_source.h"
#include "threads/pipeline.h"
#include "threads/control.h"
#include "garage/garage.h"

void init_debug() {
    auto param = Param::get_instance();
    Data::auto_fire = (*param)["Debug"]["System"]["AutoFire"];
    Data::auto_enemy = (*param)["Debug"]["System"]["AutoEnemy"];
    Data::auto_rune = (*param)["Debug"]["System"]["AutoRune"];
    Data::auto_capture = (*param)["Debug"]["System"]["AutoCapture"];

    Data::plus_pnp = (*param)["Debug"]["PlusPnP"]["Enable"];

    Data::serial_flag = (*param)["Debug"]["Control"]["Serial"];
    Data::timeout_flag = (*param)["Debug"]["Control"]["Timeout"];
    Data::manu_capture = (*param)["Debug"]["Control"]["ManuCapture"];
    Data::manu_fire = (*param)["Debug"]["Control"]["ManuFire"];

    Data::manu_rune = (*param)["Debug"]["Control"]["ManuRune"];
    Data::big_rune = (*param)["Debug"]["Control"]["BigRune"];

    Data::ui_flag = (*param)["Debug"]["ImageThread"]["UI"];
    Data::imwrite_flag = (*param)["Debug"]["ImageThread"]["Imwrite"];
    Data::image_flag = (bool)(Data::imwrite_flag || Data::imshow_flag);
    Data::binary_flag = (*param)["Debug"]["ImageThread"]["Binary"];
    Data::histogram_flag = (*param)["Debug"]["ImageThread"]["Histogram"];

    Data::reprojection_flag = (*param)["Debug"]["Display"]["Reprojection"];
    Data::pipeline_delay_flag = (*param)["Debug"]["Display"]["PipelineDelay"];
    Data::point_skip_flag = (*param)["Debug"]["Display"]["PointSkip"];

    Data::state_delay_flag = (*param)["Debug"]["StateDelay"]["Enable"];
    Data::state_delay_time = (*param)["Debug"]["StateDelay"]["TimeS"];
    Data::state_queue_size = (*param)["Debug"]["StateDelay"]["QueueSize"];
    Data::send_wait_time = (*param)["Debug"]["StateDelay"]["SendWait"];
}

bool init_camera() {
    auto param = Param::get_instance();
    auto control = Control::get_instance();

    bool is_simulation = true;
    std::string video_path = "0"; // "0" 代表使用笔记本摄像头

    if (is_simulation) {
        rm::message("Initializing Simulation Camera (Webcam/Video)...", rm::MSG_WARNING);

        Data::camera_index = 1;
        Data::camera_base = 1;

        // 读取参数配置中的宽高 (确保与模型输入一致)
        int width = (*param)["Camera"]["Laptop"]["Width"];
        int height = (*param)["Camera"]["Laptop"]["Height"];

        // 实例化 VideoSource
        Data::camera.resize(2, nullptr);
        auto* sim_cam = new VideoSource(video_path, width, height);
        Data::camera[1] = sim_cam;

        rm::mallocYoloCameraBuffer(
            &Data::camera[1]->rgb_host_buffer, 
            &Data::camera[1]->rgb_device_buffer, 
            width, 
            height
        );

        try {
            std::string camlen_path = (*param)["Camera"]["CamLensDir"];
            std::ifstream camlens_json(camlen_path);
            nlohmann::json camlens;
            camlens_json >> camlens;
            
            std::string camera_type = (*param)["Camera"]["Laptop"]["CameraType"];
            std::string lens_type = (*param)["Camera"]["Laptop"]["LensType"];
            std::vector<double> camera_offset = (*param)["Car"]["CameraOffset"]["Base"];

            Param::from_json(camlens[camera_type][lens_type]["Intrinsic"], Data::camera[1]->intrinsic_matrix);
            Param::from_json(camlens[camera_type][lens_type]["Distortion"], Data::camera[1]->distortion_coeffs);
            
            rm::tf_rotate_pnp2head(Data::camera[1]->Rotate_pnp2head, camera_offset[3], camera_offset[4], 0.0);
            rm::tf_trans_pnp2head(Data::camera[1]->Trans_pnp2head, camera_offset[0], camera_offset[1], camera_offset[2], camera_offset[3], camera_offset[4], 0.0);
            
        } catch (...) {
            rm::message("Warning: Failed to load camera intrinsics for simulation.", rm::MSG_WARNING);
        }

        // 启动采集线程
        sim_cam->startCapture();
        
        rm::message("Simulation Camera Started.", rm::MSG_OK);
        return true; // 跳过后续真实相机初始化
    }
    return true;
}

bool deinit_camera() {
    for(int i = 1; i < Data::camera.size(); i++) {
        if(Data::camera[i] == nullptr) continue;
        
        if (Data::camera[i]->rgb_host_buffer != nullptr || Data::camera[i]->rgb_device_buffer != nullptr) {
            rm::freeYoloCameraBuffer(Data::camera[i]->rgb_host_buffer, Data::camera[i]->rgb_device_buffer);
            Data::camera[i]->rgb_host_buffer = nullptr;
            Data::camera[i]->rgb_device_buffer = nullptr;
        }

        delete Data::camera[i];
        Data::camera[i] = nullptr;
        // rm::closeDaHeng();
    }
    rm::message("Camera deinit success", rm::MSG_WARNING);
    return true;
}


void init_serial() {
    int status;
    std::vector<std::string> port_list;
    auto control = Control::get_instance();

    while(true) {

        #if defined(TJURM_HERO)
        status = (int)rm::getSerialPortList(port_list, rm::SERIAL_TYPE_TTY_ACM);
        #endif

        #if defined(TJURM_BALANCE) || defined(TJURM_INFANTRY) || defined(TJURM_DRONSE) || defined(TJURM_SENTRY)
        status = (int)rm::getSerialPortList(port_list, rm::SERIAL_TYPE_TTY_USB);
        #endif

        if (status != 0 || port_list.empty()) {
            rm::message("Control port list failed", rm::MSG_ERROR);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            port_list.clear();
            continue;
        }

        control->port_name_ = port_list[0];
        status = (int)rm::openSerialPort(control->file_descriptor_, control->port_name_);
        if (status != 0) {
            rm::message("Control port open failed", rm::MSG_ERROR);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            port_list.clear();
            continue;
        }
        if(status == 0) {
            break;
        }
    }
}


void init_attack() {
    #ifdef TJURM_SENTRY
    Data::attack = new rm::Filtrate();
    #endif

    #if defined(TJURM_INFANTRY) || defined(TJURM_BALANCE) || defined(TJURM_HERO) || defined(TJURM_DRONSE)
    Data::attack = new rm::DeadLocker();
    #endif 
}