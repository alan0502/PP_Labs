CC = mpicc
CXX = mpicxx

CFLAGS = -O3 -march=native -mtune=native
CXXFLAGS = -O3 $(CFLAGS)
LDFLAGS = -flto
LDLIBS   = -lm -lnuma
TARGETS = hw1

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)

