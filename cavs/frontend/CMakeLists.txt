ADD_SUBDIRECTORY(cxx)

FILE(GLOB tmp *.cc)
SET(cxx_srcs ${cxx_srcs} ${tmp})
SET(cxx_srcs ${cxx_srcs} PARENT_SCOPE)


