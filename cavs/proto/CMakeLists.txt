# get current working directory path
execute_process(COMMAND pwd OUTPUT_VARIABLE CURRENT_WORKING_DIRECTORY)
string(STRIP ${CURRENT_WORKING_DIRECTORY} CURRENT_WORKING_DIRECTORY)

execute_process(COMMAND mkdir -p ${CURRENT_WORKING_DIRECTORY}/cavs/proto)
execute_process(COMMAND
  protoc "--cpp_out=${CURRENT_WORKING_DIRECTORY}/cavs/proto/"
  "-I${PROJECT_SOURCE_DIR}/cavs/proto"
  ${PROJECT_SOURCE_DIR}/cavs/proto/devices.proto)

execute_process(COMMAND
  protoc "--cpp_out=${CURRENT_WORKING_DIRECTORY}/cavs/proto/"
  "-I${PROJECT_SOURCE_DIR}/cavs/proto"
  ${PROJECT_SOURCE_DIR}/cavs/proto/types.proto)

execute_process(COMMAND
  protoc "--cpp_out=${CURRENT_WORKING_DIRECTORY}/cavs/proto/"
  "-I${PROJECT_SOURCE_DIR}/cavs/proto"
  ${PROJECT_SOURCE_DIR}/cavs/proto/tensor_shape.proto)

execute_process(COMMAND
  protoc "--cpp_out=${CURRENT_WORKING_DIRECTORY}/cavs/proto/"
  "-I${PROJECT_SOURCE_DIR}/cavs/proto"
  ${PROJECT_SOURCE_DIR}/cavs/proto/op_def.proto)

set(BUILD_INCLUDE ${CURRENT_WORKING_DIRECTORY} PARENT_SCOPE)
