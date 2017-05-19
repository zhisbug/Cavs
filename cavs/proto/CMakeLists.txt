FILE(GLOB proto_files "*.proto")

IF (FALSE)
  MESSAGE(STATUS ${proto_files})
ENDIF()

FUNCTION(protobuf_gen_cpp gen_header_files gen_cpp_files)
  IF (NOT ARGN)
    MESSAGE(FATAL_ERROR "No Proto Files")
    RETURN()
  ENDIF()

  SET(${gen_header_files})
  SET(${gen_cpp_files})
  SET(PROTOC_COMMAND "protoc")
  SET(CMAKE_COMMAND "cmake")
  FOREACH(f ${ARGN})
    GET_FILENAME_COMPONENT(f      ${f} ABSOLUTE)
    GET_FILENAME_COMPONENT(f_name ${f} NAME_WE)

    SET(header  "${CMAKE_CURRENT_BINARY_DIR}/${f_name}.pb.h" )
    SET(cpp     "${CMAKE_CURRENT_BINARY_DIR}/${f_name}.pb.cc")

    LIST(APPEND headers ${header})
    LIST(APPEND cpps    ${cpp}   )

    ADD_CUSTOM_COMMAND(
      OUTPUT ${header} ${cpp} 
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND ${PROTOC_COMMAND} -I${PROJECT_SOURCE_DIR} --cpp_out ${PROJECT_BINARY_DIR} ${f}
      COMMENT "Compiling Protobuf files into header and cc files"
      DEPENDS ${f}
      VERBATIM
      )
  ENDFOREACH()

  SET_SOURCE_FILES_PROPERTIES(${headers} ${cpps} PROPERTIES GENERATED TRUE)
  SET(${gen_header_files} ${headers} PARENT_SCOPE)
  SET(${gen_cpp_files}    ${cpps}    PARENT_SCOPE)
ENDFUNCTION()

protobuf_gen_cpp(gen_headers gen_cpps ${proto_files})
ADD_LIBRARY(proto_def OBJECT ${gen_cpps} ${gen_headers})
