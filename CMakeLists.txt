cmake_minimum_required(VERSION 3.2)
project(machineLearning)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(ML ${SRC})