ADD_SUBDIRECTORY(proto)
ADD_SUBDIRECTORY(midend)
ADD_SUBDIRECTORY(frontend)

MESSAGE(STATUS ${cxx_srcs})
MESSAGE(STATUS ${cuda_srcs})

ADD_LIBRARY(cavs_cxx ${cxx_srcs} $<TARGET_OBJECTS:proto_def>)
ADD_LIBRARY(cavs_cuda ${cuda_srcs} $<TARGET_OBJECTS:proto_def>)
