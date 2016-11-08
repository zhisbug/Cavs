NVCC=/usr/local/cuda/bin/nvcc#--verbose
CXX=g++
CXXFLAGS=-O2 -std=c++11 -I/usr/local/cuda/include
NVFLAGS=$(CXXFLAGS) -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52
LDFLAGS=-L/usr/local/cuda/lib64 -lcudart  -lcublas -lgflags -lglog 


ALL: test_main
test_main: test_main.o functions.cuo
	$(CXX) $(LDFLAGS) -o $@ $^
test_main.o: test_main.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $^
functions.cuo: functions.cu
	$(NVCC) $(NVFLAGS) -o $@ -c $^
clean:
	rm *.o *.cuo test_main
