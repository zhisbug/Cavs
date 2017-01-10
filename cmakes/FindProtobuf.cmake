find_path(Protobuf_INCLUDE_DIR google/protobuf/service.h
  NO_DEFAULT_PATH
  PATHS
  "/usr/local/include"
  "/usr/include")

find_library(Protobuf_LIBRARY
  NAMES protobuf libprotobuf
  HINTS 
  "/usr/local/lib"
  "/usr/lib")

message(STATUS "Find Protobuf include path: ${Protobuf_INCLUDE_DIR}")
message(STATUS "Find Protobuf lib path: ${Protobuf_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Protobuf
  REQUIRED_VARS
  Protobuf_LIBRARY
  Protobuf_INCLUDE_DIR)

set(Protobuf_LIBRARIES ${Protobuf_LIBRARY})

mark_as_advanced(
  Protobuf_LIBRARIES
  Protobuf_INCLUDE_DIR)
