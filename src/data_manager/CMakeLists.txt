set(HIK_LIB_PATH "/opt/MVS/lib/64") # TODO 注意 可能需要根据海康SDK安装路径修改
find_library(HIK_MV_LIB       NAMES MvCameraControl   PATHS ${HIK_LIB_PATH})
find_library(HIK_CONVERT_LIB  NAMES FormatConversion  PATHS ${HIK_LIB_PATH})
find_library(HIK_RENDER_LIB   NAMES MVRender          PATHS ${HIK_LIB_PATH})
find_library(HIK_USB_LIB      NAMES MvUsb3vTL         PATHS ${HIK_LIB_PATH})
find_library(HIK_MEDIA_LIB    NAMES MediaProcess      PATHS ${HIK_LIB_PATH}) # 推荐添加

file(GLOB_RECURSE
    data_src 
        ${CMAKE_SOURCE_DIR}/src/data_manager/*.cpp
)

set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(${CMAKE_SOURCE_DIR}/libs/json json_binary_dir)
set(JSON_LIBS 
        nlohmann_json::nlohmann_json
        # -lgxiapi
        ${HIK_MV_LIB} 
        ${HIK_CONVERT_LIB} 
        ${HIK_RENDER_LIB} 
        ${HIK_USB_LIB} 
        ${HIK_MEDIA_LIB} 
        -lpthread
        -ldl
)


add_library(data_manager
    STATIC
        ${data_src}    
)

target_link_libraries(data_manager
    PUBLIC
        ${OpenCV_LIBS}
        ${OpenRM_LIBS}
        ${JSON_LIBS}
)
