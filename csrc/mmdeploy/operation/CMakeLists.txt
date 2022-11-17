# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_operation)

mmdeploy_add_module(${PROJECT_NAME} LIBRARY operations.cpp)

add_subdirectory(cpu)
add_subdirectory(cuda)

add_library(mmdeploy::operation ALIAS ${PROJECT_NAME})
