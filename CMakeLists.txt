cmake_minimum_required(VERSION 2.8.3)
project(stereo_detector)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  geometry_msgs
  message_generation
  message_runtime
)


add_message_files(
  FILES
    BoundingBox.msg
    BoundingBoxes.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
    sensor_msgs  
    std_msgs
)


catkin_package(CATKIN_DEPENDS 
  roscpp std_msgs 
  geometry_msgs 
  sensor_msgs 
  message_runtime 
  roslib
)

###########
## Build ##
###########


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
)


#
# CUDA Configuration
#
find_package(CUDA 10.0 REQUIRED)

set(CUDA_VERBOSE_BUILD ON)

# Specify the cuda host compiler to use the same compiler as cmake.
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
  find_library(TENSORRT_LIBRARY_PARSER nvparsers
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR
    "Cannot find TensorRT library.")
endif()

FIND_PACKAGE(OpenCV 3.2.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")	# -std=gnu++11
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")

list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")
CUDA_INCLUDE_DIRECTORIES(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} 
  Detector.cpp 
  main.cpp 
  StereoDetectorNode.cpp 
  TensorRT.cpp 
  Utils.cpp 
  EntropyCalibrator.cpp
  SGM.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} 
  ${TENSORRT_LIBRARY} 
  ${OpenCV_LIBS} 
  nvonnxparser 
  ${catkin_LIBRARIES}
  sgm
  boost_thread
  boost_system)

# add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_generate_messages_cpp)
