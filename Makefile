CXX=g++
CXXFLAGS=-O2 -std=c++11 -I. -Ibuild/ -I/usr/local/cuda/include
PROTOCC=protoc
PROTOFLAGS=--cpp_out=build/ -I. 
NVCC=/usr/local/cuda/bin/nvcc
NVFLAGS=$(CXXFLAGS) -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 --default-stream per-thread
LDFLAGS=-L/usr/local/lib -lprotobuf -lglog -lgflags -L/usr/local/cuda/lib64 -lcudart  -lcublas -lcudnn -pthread

C_SRC=$(filter-out %_test.cc, $(wildcard cavs/*/*.cc cavs/*/*/*.cc))
C_OBJS = $(patsubst cavs/%.cc, build/cavs/%.o, $(C_SRC))
CU_SRC=$(filter-out %_test.cu, $(wildcard cavs/*/*.cu ))
CU_OBJS = $(patsubst cavs/%.cu, build/cavs/%.cuo, $(CU_SRC))
TEST_SRC=$(wildcard cavs/*/*_test.c* cavs/*/*/*_test.c*)
APP_SRC=$(wildcard apps/topic_model_mf/*.cc apps/lenet-5/*)
PROTO = $(wildcard cavs/*/*.proto)
PROTO_HEADERS = $(patsubst cavs/%.proto, build/cavs/%.pb.h, $(PROTO))
PROTO_SRCS = $(patsubst cavs/%.proto, build/cavs/%.pb.cc, $(PROTO))
PROTO_OBJS = $(patsubst cavs/%.proto, build/cavs/%.pb.o, $(PROTO))

LIB=lib/libcavs.a
TEST_BIN=$(patsubst cavs/%.cc, build/test/%, $(TEST_SRC))
TEST_BIN+=$(patsubst cavs/%.cu, build/test/%, $(TEST_SRC))
APP_BIN=$(patsubst apps/%.cc, build/apps/%, $(APP_SRC))

.PHONY: clean

all: $(PROTO_SRCS) $(LIB) $(TEST_BIN) $(APP_BIN)

.PRECIOUS: $(PROTO_SRCS) $(PROTO_HEADERS)

$(LIB): $(C_OBJS) $(CU_OBJS) $(PROTO_OBJS) 
	mkdir -p $(@D)
	ar crv $@ $^
build/cavs/%.pb.o: cavs/%.pb.cc
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@ 
build/cavs/%.pb.cc: cavs/%.proto
	mkdir -p $(@D)
	$(PROTOCC) $(PROTOFLAGS) $<
build/cavs/%.o: cavs/%.cc 
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@ 
build/cavs/%.cuo: cavs/%.cu 
	mkdir -p $(@D)
	$(NVCC) $(NVFLAGS) -c $< -o $@ 
build/test/%.cuo: cavs/%.cu 
	mkdir -p $(@D)
	$(NVCC) $(NVFLAGS) -c $< -o $@ 
build/test/% : build/cavs/%.o $(LIB)
	mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -o $@ $< -Wl,--whole-archive $(LIB) -Wl,--no-whole-archive
build/test/% : build/cavs/%.cuo $(LIB)
	mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -o $@ $< -Wl,--whole-archive $(LIB) -Wl,--no-whole-archive
build/apps/%.o: apps/%.cc 
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@ 
build/apps/% : build/apps/%.o $(LIB)
	mkdir -p $(@D)
	$(CXX) -o $@ $< -Wl,--whole-archive $(LIB) -Wl,--no-whole-archive $(LDFLAGS)
clean:
	rm -rf build/ lib/

