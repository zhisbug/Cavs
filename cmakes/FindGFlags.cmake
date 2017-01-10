find_path(GFlags_INCLUDE_DIR gflags/gflags.h
  NO_DEFAULT_PATH
  PATHS
  "/usr/local/include"
  "/usr/include")

find_library(GFlags_LIBRARY
  NAMES gflags libgflags
  HINTS 
  "/usr/local/lib"
  "/usr/lib")

message(STATUS "Find GFlags include path: ${GFlags_INCLUDE_DIR}")
message(STATUS "Find GFlags lib path: ${GFlags_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  GFlags
  REQUIRED_VARS
  GFlags_LIBRARY
  GFlags_INCLUDE_DIR)

set(GFlags_LIBRARIES ${GFlags_LIBRARY})

mark_as_advanced(
  GFlags_LIBRARIES
  GFlags_INCLUDE_DIR)
