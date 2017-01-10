find_path(Cudnn_INCLUDE_DIR cudnn.h
  NO_DEFAULT_PATH
  PATHS
  "/usr/local/cuda/include"
  "/usr/include"
  "/usr/local/include")

find_library(Cudnn_LIBRARY
  NAMES cudnn libcudnn 
  HINTS
  "/usr/local/cuda/lib"
  "/usr/lib"
  "/usr/local/lib")

message(STATUS "Find Cudnn include path: ${Cudnn_INCLUDE_DIR}")
message(STATUS "Find Cudnn lib path: ${Cudnn_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Cudnn
  REQUIRED_VARS
  Cudnn_LIBRARY
  Cudnn_INCLUDE_DIR)

set(Cudnn_LIBRARIES ${Cudnn_LIBRARY})
mark_as_advanced(
  Cudnn_LIBRARIES
  Cudnn_INCLUDE_DIR)
