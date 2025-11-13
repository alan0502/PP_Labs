CC = gcc
CXX = g++

CXXFLAGS = -O2 -pthread -fopenmp
LDFLAGS = -lm
EXES = hw3-1

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

hw3-1: hw3-1.cc
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $?
