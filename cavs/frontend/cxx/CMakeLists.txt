FILE(GLOB cxx_srcs *.cc)
FILE(GLOB test_srcs *_test.cc)

FOREACH(test ${test_srcs})
  LIST(REMOVE_ITEM cxx_srcs ${test})
ENDFOREACH()

SET(cxx_srcs ${cxx_srcs} PARENT_SCOPE)
