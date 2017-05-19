ADD_SUBDIRECTORY(cxx)

FILE(GLOB curr_cxx_srcs *.cc)
SET(cxx_srcs ${cxx_srcs} ${curr_cxx_srcs} PARENT_SCOPE)
