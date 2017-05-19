ADD_SUBDIRECTORY(proto)
ADD_SUBDIRECTORY(frontend)
ADD_SUBDIRECTORY(midend)
#ADD_SUBDIRECTORY(backend)

IF(TRUE)
  MESSAGE(STATUS ${cxx_srcs}  "[CXX]")
  MESSAGE(STATUS ${cuda_srcs} "[CUDA]")
ENDIF()

ADD_LIBRARY(cavs_cxx ${cxx_srcs} $<TARGET_OBJECTS:proto_def>)
CUDA_ADD_LIBRARY(cavs_cuda ${cuda_srcs} $<TARGET_OBJECTS:proto_def>)
