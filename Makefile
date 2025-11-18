CC = g++
INCLUDE_DIRS = ./hexl-bindings/hexl/hexl/include/
FLAGS = -Wall -Wno-unused-function -Wno-unused-result -funroll-all-loops -march=native -lm -Wno-sign-compare -Wno-write-strings
FLAGS += -mavx512f -mavx512dq

OPT_FLAGS = -O3 -fwhole-program -flto $(FLAGS)
DEBUG_FLAGS = -O0 -g $(FLAGS)
LD_LIBS = hexl
LIB_DIRS = ./hexl-bindings/hexl/build/hexl/lib ./hexl-bindings/hexl/build/hexl/lib64
LIBS += $(addprefix -L, $(LIB_DIRS)) $(addprefix -l, $(LD_LIBS))
INCLUDE_FLAGS = $(addprefix -I, $(INCLUDE_DIRS))
FLAGS += $(INCLUDE_FLAGS)
LIB_FLAGS = -O3 $(FLAGS)


SRC=polynomial.cpp misc.cpp

ALL_SRC = $(addprefix ./, $(SRC))

wrapper:
	$(CC) -std=c++17 -fPIC -shared -o libhexl_wrapper.so hexl-bindings/hexl_wrapper.cpp $(OPT_FLAGS) $(LIBS)

hexl: hexl/build
	cmake --build ./hexl-bindings/hexl/build

hexl/build:
	cmake -S ./hexl-bindings/hexl/ -B ./hexl-bindings/hexl/build -DCMAKE_POLICY_VERSION_MINIMUM=3.5

hexl-triton: hexl/build-triton
	cmake --build ./hexl-bindings/hexl/build

hexl/build-triton:
	cmake -S ./hexl-bindings/hexl/ -B ./hexl-bindings/hexl/build -DCMAKE_C_COMPILER=/appl/scibuilder-spack/aalto-rhel9-prod/2024-01-compilers/software/linux-rhel9-haswell/gcc-11.4.1/gcc-12.3.0-xh5vv5d/bin/gcc

# hexl/prepare:
#	module load cmake gcc 

# Ensure AVX-512 support for _mm512_rem_epi64
