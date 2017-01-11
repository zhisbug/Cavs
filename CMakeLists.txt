cmake_minimum_required(VERSION 2.8.9)

# find gcc/g++
if(NOT "${CAVS_SYSTEM_NAME}" MATCHES "Darwin")
  find_program(CMAKE_CXX_COMPILER NAMES g++)
  find_program(CMAKE_C_COMPILER NAMES gcc)
endif()

project(Cavs CXX C)

# CXXFLAGS for gcc
if(CMAKE_COMPILE_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -O2")
  set(CMAKE_CXX_FLAGS_DEBUG "-Wall -std=c++11 -O0 -g -fno-inline")
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O3 -DNDEBUG -march=native -ffast-math")
endif()

# CXXFLAGS for clang
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -O2")
  set(CMAKE_CXX_FLAGS_DEBUG "-Wall -std=c++11 -O0 -g -fno-inline")
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O3 -DNDEBUG -march=native -ffast-math")
endif()
if(CMAKE_BUILD_TYPE MATCHES RELEASE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
endif()

# local include
include_directories(${PROJECT_SOURCE_DIR})

# third party
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmakes")

find_package(GFlags REQUIRED)
include_directories(${GFlags_INCLUDE_DIR})

find_package(Glog REQUIRED)
include_directories(${Glog_INCLUDE_DIR})

find_package(CUDA 7.5 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
# TODO
set(CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52")

find_package(Cudnn REQUIRED)
include_directories(${Cudnn_INCLUDE_DIR})

link_libraries(${PROTOBUF_LIBRARIES} ${GFlags_LIBRARIES} ${Glog_LIBRARIES} ${CUDA_LIBRARIES} ${Cudnn_LIBRARIES})

# output
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/lib")

# subdirectory
add_subdirectory(cavs/proto)
include_directories(${BUILD_INCLUDE})

add_subdirectory(cavs/frontend)
add_subdirectory(cavs/midend)
add_subdirectory(cavs/backend)

# debug purpose
#get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#foreach(dir ${dirs})
#  message(STATUS "dir='${dir}'")
#endforeach()
