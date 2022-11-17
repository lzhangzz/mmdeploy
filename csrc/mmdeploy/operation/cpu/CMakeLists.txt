# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_operation_cpu)

add_library(${PROJECT_NAME} OBJECT
        resize.cpp
        cvtcolor.cpp
        pad.cpp
        to_float.cpp
        hwc2chw.cpp
        normalize.cpp
        crop.cpp
        flip.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_link_libraries(${PROJECT_NAME} PRIVATE
        mmdeploy_core
        mmdeploy_opencv_utils)
target_link_libraries(mmdeploy_operation PRIVATE
        ${PROJECT_NAME})
mmdeploy_export(${PROJECT_NAME})
