ADD_SUBDIRECTORY(cxx)

FILE(GLOB curr_cxx_srcs *.cc)
SET(cxx_srcs ${cxx_srcs} ${curr_cxx_srcs} PARENT_SCOPE)
SET(test_cxx_srcs ${test_cxx_srcs} ${curr_test_srcs} PARENT_SCOPE)
