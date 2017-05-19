cmake_minimum_required(VERSION 2.8)
project(Cavs CXX C)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(BEFORE ${PROJECT_BINARY_DIR})

set(EXECUTABLE_OUTPUT_PATH, "${PROJECT_SOURCE_DIR/bin}")
set(LIBRARY_OUTPUT_PATH, "${PROJECT_SOURCE_DIR/lib}")
add_subdirectory(cavs)
#add_subdirectory(apps)
