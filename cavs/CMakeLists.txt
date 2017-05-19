#SET(cxx_srcs)

ADD_SUBDIRECTORY(proto)
#ADD_SUBDIRECTORY(midend)
message(STATUS, ${cxx_srcs})

message(STATUS, ${proto_srcs})

#add_library(Cavs $<TARGET_OBJECTS:proto_srcs>)
