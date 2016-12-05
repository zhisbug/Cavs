#CC=gcc
CXX=g++
FLAGS=
LDFLAGS=

ALL: topic_model_mf

topic_model_mf: topic_model_mf.o paras.o utils.o tm_mf.o
	$(CXX) $(LDFLAGS) $^ -o $@ 
%.o: %.cpp
	$(CXX) $(FLAGS) -c $^ -o $@

clean:
	rm *.o topic_model_mf
