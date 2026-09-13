UNAME_S := $(shell uname -s)

# OMPFLAG isolado do resto de CXXFLAGS para poder ligar/desligar so o OpenMP
# pela linha de comando (o binario sequencial de referencia e o MESMO
# codigo-fonte, so compilado sem esta flag -- nao um train.cpp separado):
#   make all                  -> paralelo (com -fopenmp)
#   make all OMPFLAG=         -> sequencial de referencia (sem -fopenmp)
OMPFLAG = -fopenmp

ifeq ($(UNAME_S),Darwin)
# macOS: g++/gcc do sistema e Apple Clang (sem suporte a -fopenmp) -> usa o
# GCC de verdade do Homebrew (brew install gcc) e aponta o sysroot para o
# SDK do Xcode/Command Line Tools, senao o GCC do Homebrew nao acha os
# headers padrao (math.h, stdio.h etc).
CXX = g++-16
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra $(OMPFLAG) -isysroot $(shell xcrun --show-sdk-path)
else
# Linux: g++ do sistema ja suporta -fopenmp direto.
CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra $(OMPFLAG)
endif

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
