FILE(GLOB tmp *.cc)
SET(cxx_srcs ${cxx_srcs} ${tmp})
message(STATUS, ${cxx_srcs})

set(cxx_srcs ${cxx_srcs} PARENT_SCOPE)

#add_library(allocator OBJECT allocator.cc)
#add_library(devices OBJECT devices.cc)
#add_library(op_context OBJECT op_context.cc)
#add_library(session OBJECT session.cc)
#add_library(tensor OBJECT tensor.cc)

#cuda_add_library(cavs_allocator allocator.cu)
#cuda_add_library(cavs_device device.cu)
