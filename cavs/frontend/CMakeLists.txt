add_library(c_api OBJECT c_api.cc)
add_library(dep_graph OBJECT dep_graph.cc)
add_library(node OBJECT node.cc)

add_subdirectory(cxx)
