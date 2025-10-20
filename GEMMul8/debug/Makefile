
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')

CUDA_PATH ?= /usr/local/cuda
export PATH := $(CUDA_PATH)/bin:$(PATH)
export LD_LIBRARY_PATH := $(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH)

COMPILER := nvcc
LIBS := -lcublas -lcudart -lcuda -lnvidia-ml -ldl ../lib/libgemmul8.a
FLAGS := -std=c++20 -O3 -I../include
ARCH := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

TARGET := test

all: $(TARGET)

$(TARGET): $(TARGET).cu
	$(COMPILER) $< $(FLAGS) $(ARCH) -o $@ $(LIBS)

run:
	compute-sanitizer --tool memcheck ./$(TARGET)

clean:
	rm -f *.o
	rm -f $(TARGET)
