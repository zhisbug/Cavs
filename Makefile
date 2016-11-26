CXX=g++
CXXFLAGS=-O2 -std=c++11 -I. -Ibuild/ -I/usr/local/cuda/include
PROTOCC=protoc
PROTOFLAGS=--cpp_out=build/ -I. 
NVCC=/usr/local/cuda/bin/nvcc#--verbose
NVFLAGS=$(CXXFLAGS) -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52
LDFLAGS=-lgflags -lglog -L/usr/local/lib -lprotobuf -L/usr/local/cuda/lib64 -lcudart  -lcublas 

C_SRC=$(filter-out %_test.cc, $(wildcard cavs/*/*.cc ))
C_OBJS = $(patsubst cavs/%.cc, build/cavs/%.o, $(C_SRC))
CU_SRC=$(filter-out %_test.cu, $(wildcard cavs/*/*.cu ))
CU_OBJS = $(patsubst cavs/%.cu, build/cavs/%.cuo, $(CU_SRC))
TEST_SRC=$(wildcard cavs/*/*_test.cc )
#DEPS= $(SRCS:.cc=.d)
PROTO = $(wildcard cavs/*/*.proto)
PROTO_HEADERS = $(patsubst cavs/%.proto, build/cavs/%.pb.h, $(PROTO))
PROTO_SRCS = $(patsubst cavs/%.proto, build/cavs/%.pb.cc, $(PROTO))
PROTO_OBJS = $(patsubst cavs/%.proto, build/cavs/%.pb.o, $(PROTO))

LIB=lib/libcavs.a
TEST_BIN=$(patsubst cavs/%.cc, build/test/%, $(TEST_SRC))

.PHONY: clean

all: $(PROTO_SRCS) $(LIB) $(TEST_BIN)

.PRECIOUS: $(PROTO_SRCS) $(PROTO_HEADERS)

$(LIB): $(C_OBJS) $(CU_OBJS) $(PROTO_OBJS) 
	mkdir -p $(@D)
	ar crv $@ $^
#build/cavs/%.d: cavs/%.cc 
	#mkdir -p $(@D)
	#$(CXX) $(CXXFLAGS) -MM -MT build/$*.o $< >build/$*.d
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
build/test/% : build/cavs/%.o 
	mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -o $@ $< -Wl,--whole-archive $(LIB) -Wl,--no-whole-archive
clean:
	rm -rf build/ lib/

#/usr/local/cuda/bin/nvcc -ccbin c++ -std=c++11 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -O2 -Xcompiler "-fPIC -ffast-math -pthread -Wall -Wextra -Wno-sign-compare -ffunction-sections -fdata-sections -O2" -I/home/deeplearning/caffe2/gen -I/home/deeplearning/caffe2/gen/third_party -I/home/deeplearning/caffe2/gen/third_party/include -I/usr/include/openmpi-x86_64 -I/usr/local/cuda/include -DNDEBUG -DGTEST_USE_OWN_TR1_TUPLE=1 -DEIGEN_NO_DEBUG -DPIC -DCAFFE2_BUILD_STRING=\\"971eee9,with_local_changes\\" -DCAFFE2_USE_EIGEN_FOR_BLAS -DCAFFE2_USE_GOOGLE_GLOG -DCAFFE2_FORCE_FALLBACK_CUDA_MPI -c caffe2/utils/math_gpu.cu -o /home/deeplearning/caffe2/gen/caffe2/utils/math_gpu.cuo
