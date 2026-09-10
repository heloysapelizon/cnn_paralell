CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra

CPP_DIR = cpp
PY_DIR = python

COMMON_SRCS = $(CPP_DIR)/layers.cpp $(CPP_DIR)/network.cpp
TRAIN_SRCS = $(COMMON_SRCS) $(CPP_DIR)/mnist.cpp $(CPP_DIR)/train.cpp

.PHONY: all clean test

all: train

train: $(TRAIN_SRCS) $(CPP_DIR)/layers.hpp $(CPP_DIR)/network.hpp $(CPP_DIR)/mnist.hpp
	$(CXX) $(CXXFLAGS) -o train $(TRAIN_SRCS)

test:
	python3 $(PY_DIR)/gradient_check.py

clean:
	rm -f train *.o
