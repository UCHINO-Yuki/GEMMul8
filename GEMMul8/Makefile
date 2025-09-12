#===============
# path to cuda
#===============
CUDA_PATH := /usr/local/cuda


export PATH := $(CUDA_PATH)/bin:$(PATH)
export PATH := $(CUDA_PATH)/bin/lib64:$(PATH)

GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
ARCHS := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)
NVCCFLAGS := -std=c++20 $(ARCHS) -lnvidia-ml -O3 -DGEMMul8_ARCH=$(GPU_ARCH)
ifneq ($(THREADS1),)
NVCCFLAGS += -DTHREADS1=$(THREADS1)
endif
ifneq ($(THREADS2),)
NVCCFLAGS += -DTHREADS2=$(THREADS2)
endif
ifneq ($(THREADS3),)
NVCCFLAGS += -DTHREADS3=$(THREADS3)
endif
LIBS := -lcublas -lcudart -lcuda -ldl 


#===============
# static
#===============
LIB_TARGET := lib/libgemmul8.a
HEADERS := include/gemmul8.hpp
CU_FILES := src/gemmul8.cu
CU_OBJS := $(CU_FILES:.cu=.o)

all: $(LIB_TARGET) hook

$(LIB_TARGET): $(CU_OBJS)
	mkdir -p lib
	ar rcs $@ $^
	cd testing ; make ;cd ../ 

%.o: %.cu $(HEADERS)
	nvcc $(NVCCFLAGS) $(LIBS) -c $< -o $@

#===============
# shared
#===============
HOOK_TARGET := lib/libgemmul8.so
HOOK_SRC    := src/hook.cu
GEMMUL8_SHARED_OBJ := gemmul8_shared.o
NVCCFLAGS_SHARED := $(NVCCFLAGS) -Xcompiler -fPIC

$(GEMMUL8_SHARED_OBJ): src/gemmul8.cu $(HEADERS)
	nvcc $(NVCCFLAGS_SHARED) -c src/gemmul8.cu -o $(GEMMUL8_SHARED_OBJ)
	
hook: $(GEMMUL8_SHARED_OBJ)
	mkdir -p lib
	nvcc $(NVCCFLAGS_SHARED) -c $(HOOK_SRC) -o src/hook.o
	nvcc $(NVCCFLAGS_SHARED) -shared -o $(HOOK_TARGET) $(GEMMUL8_SHARED_OBJ) src/hook.o $(LIBS)

clean:
	cd testing ; make clean ;cd ../ 
	rm -f $(CU_OBJS) $(LIB_TARGET) src/hook.o $(HOOK_TARGET)

