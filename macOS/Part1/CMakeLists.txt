cmake_minimum_required(VERSION 3.6)
project(Part1)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(Part1 ${SOURCE_FILES})

target_link_libraries(Part1 ${OpenCV_LIBS})