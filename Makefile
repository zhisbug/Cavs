CXX=g++
CXXFLAGS=-O2 -std=c++11 -I/usr/local/cuda/include
PROTOCC=protoc
#PROTOFLAGS=--cpp_out=./
NVCC=/usr/local/cuda/bin/nvcc#--verbose
NVFLAGS=$(CXXFLAGS) -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52
LDFLAGS=-L/usr/local/cuda/lib64 -lcudart  -lcublas -lgflags -lglog -L/usr/local/lib -lprotobuf

SRC=$(filter-out %_test.cc, $(wildcard src/*/*.cc ))
SRC_OBJS = $(patsubst src/%.cc, build/%.o, $(SRC))
TEST_SRC=$(wildcard src/*/*_test.cc )
DEPS= $(SRCS:.cc=.d)
PROTO = $(wildcard src/*/*.proto)
PROTO_HEADERS = $(patsubst src/%.proto, build/%.pb.h, $(PROTO))
PROTO_SRCS = $(patsubst src/%.proto, build/%.pb.cc, $(PROTO))
PROTO_OBJS = $(patsubst src/%.proto, build/%.pb.o, $(PROTO))

LIB=lib/libcavs.a
TEST_BIN=$(patsubst src/%.cc, build/test/%, $(TEST_SRC))

.PHONY: clean

all: $(PROTO_SRCS) $(LIB) $(TEST_BIN)

.PRECIOUS: $(PROTO_SRCS) $(PROTO_HEADERS)

$(LIB): $(SRC_OBJS) $(PROTO_OBJS) 
	mkdir -p $(@D)
	ar crv $@ $^
build/%.d: src/%.cc 
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MM -MT build/$*.o $< >build/$*.d
build/%.pb.o: src/%.pb.cc
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I$(@D)
build/%.pb.cc: src/%.proto
	mkdir -p $(@D)
	$(PROTOCC) --proto_path=src/core --cpp_out=build/core $<
build/%.o: src/%.cc 
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I$(@D)
build/test/% : build/%.o $(LIB)
	mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -o $@ $^

clean:
	rm -rf build/ lib/
