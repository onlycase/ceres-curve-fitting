cmake_minimum_required(VERSION 3.18.1)
project(ceres-curve-fitting)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIRS})

set(SOURCE_FILES src/main.cc)
add_executable(ceres-curve-fitting ${SOURCE_FILES})
target_link_libraries(ceres-curve-fitting ${CERES_LIBRARIES} ${OpenCV_LIBS})
