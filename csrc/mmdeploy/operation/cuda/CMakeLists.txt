if (NOT ("cuda" IN_LIST MMDEPLOY_TARGET_DEVICES))
    return()
endif ()

project(mmdeploy_operation_cuda CUDA CXX)

find_package(pplcv REQUIRED)

add_library(${PROJECT_NAME} OBJECT
        resize.cpp
        cvtcolor.cpp
        pad.cpp
        to_float.cpp
        cast.cu
        hwc2chw.cpp
        transpose.cu
        normalize.cpp
        normalize.cu
        crop.cpp
        crop.cu
        flip.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_link_libraries(${PROJECT_NAME} PRIVATE
        mmdeploy_core
        ${PPLCV_LIBRARIES})
target_include_directories(${PROJECT_NAME}
        PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include ${PPLCV_INCLUDE_DIRS})
target_link_libraries(mmdeploy_operation PRIVATE
        ${PROJECT_NAME})
mmdeploy_export(${PROJECT_NAME})
