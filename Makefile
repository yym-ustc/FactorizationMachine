CXX = /specific_path/gcc-release-4.8.5/bin/g++
CXXFLAGS = -Wall -O3 -std=c++0x -march=corei7-avx

# comment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
CXXFLAGS += -fopenmp

all: fm-train fm-predict

fm-train: fm-train.cpp fm.o timer.o
	$(CXX) $(CXXFLAGS) $(DFLAG) -o $@ $^

fm-predict: fm-predict.cpp fm.o timer.o
	$(CXX) $(CXXFLAGS) $(DFLAG) -o $@ $^

fm.o: fm.cpp fm.h timer.o
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

timer.o: timer.cpp timer.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

clean:
	rm -f fm-train fm-predict fm.o timer.o
