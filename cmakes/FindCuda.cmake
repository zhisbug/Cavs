find_path(Cuda_INCLUDE_DIR cuda.h
  NO_DEFAULT_PATH
  PATHS
  "/usr/local/cuda/include"
  "/usr/include"
  "/usr/local/include")

find_library(Cuda_LIBRARY
  NAMES cudart libcudart cublas libcublas
  HINTS
  "/usr/local/cuda/lib"
  "/usr/lib"
  "/usr/local/lib")

message(STATUS "Find Cuda include path: ${Cuda_INCLUDE_DIR}")
message(STATUS "Find Cuda lib path: ${Cuda_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Cuda
  REQUIRED_VARS
  Cuda_LIBRARY
  Cuda_INCLUDE_DIR)

set(Cuda_LIBRARIES ${Cuda_LIBRARY})
mark_as_advanced(
  Cuda_LIBRARIES
  Cuda_INCLUDE_DIR)
