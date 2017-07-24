FILE(GLOB curr_cxx_srcs *.cc)
FILE(GLOB curr_test_srcs *_test.cc)
FOREACH(test ${curr_test_srcs})
  LIST(REMOVE_ITEM curr_cxx_srcs ${test})
ENDFOREACH()

SET(cxx_srcs ${cxx_srcs} ${curr_cxx_srcs} PARENT_SCOPE)
SET(test_cxx_srcs ${test_cxx_srcs} ${curr_test_srcs} PARENT_SCOPE)
