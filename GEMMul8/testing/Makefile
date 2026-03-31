#===============
# options
#===============

CUDA_PATH ?= /usr/local/cuda
HIP_PATH ?= /opt/rocm
BACKEND ?= auto
GPU_ARCH ?= auto
GPU_MEM_MB ?= auto
INFO_FLAG ?= yes


#===============
# Auto-detect backend (CUDA or HIP)
#===============

ifeq ($(BACKEND),auto)
ifneq ($(shell command -v nvidia-smi 2>/dev/null),)
override BACKEND := cuda
else ifneq ($(shell command -v rocminfo 2>/dev/null),)
override BACKEND := hip
else ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),run)
$(error Neither NVIDIA (CUDA) nor AMD (ROCm) GPU environment detected!)
endif
endif
endif


#===============
# CUDA setup
#===============

ifeq ($(BACKEND),cuda)

ifeq ($(GPU_ARCH),auto)
override GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
endif

ifeq ($(GPU_MEM_MB),auto)
override GPU_MEM_MB := $(shell nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),run)
ifeq ($(shell echo "$(GPU_MEM_MB)" | grep -E '^[0-9]+$$'),)
$(error GPU_MEM_MB could not be detected)
endif
endif
endif
endif

export PATH := $(CUDA_PATH)/bin:$(PATH)
export LD_LIBRARY_PATH := $(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH)

COMPILER := nvcc
LIBS := -lcublas -lcublasLt -lcudart -lcuda -lnvidia-ml -ldl ../lib/libgemmul8.a
FLAGS := -std=c++20 -O3 -I../include
FLAGS += -DGPU_MEM_MB=$(GPU_MEM_MB)
ARCH := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

endif


#===============
# HIP setup
#===============

ifeq ($(BACKEND),hip)

ifeq ($(GPU_ARCH),auto)
override GPU_ARCH := $(shell amd-smi static --asic --csv | grep -oE 'gfx[0-9]+[a-z]*' | head -n 1)
endif

export PATH := $(HIP_PATH)/bin:$(PATH)
export LD_LIBRARY_PATH := $(HIP_PATH)/lib:$(LD_LIBRARY_PATH)

ifeq ($(GPU_MEM_MB),auto)
override GPU_MEM_MB := $(shell amd-smi static --vram --csv | awk -F, "NR==2 {print \$$4}")
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),run)
ifeq ($(shell echo "$(GPU_MEM_MB)" | grep -E '^[0-9]+$$'),)
$(error GPU_MEM_MB could not be detected)
endif
endif
endif
endif

COMPILER := hipcc
LIBS := -lamd_smi -lamdhip64 -lhipblas -lhipblaslt -ldl -Wl,../lib/libgemmul8.a
FLAGS := -std=c++20 -O3 -I../include
FLAGS += -ffp-contract=off
FLAGS += -Wno-unused-result -Wno-unused-command-line-argument -Wno-unused-value
FLAGS += -DOCML_BASIC_ROUNDED_OPERATIONS
FLAGS += -DGPU_MEM_MB=$(GPU_MEM_MB)
ARCH := --offload-arch=$(GPU_ARCH)

endif


#===============
# Compile
#===============

TARGET := test

all: INFO VERSION $(TARGET)

INFO:
ifeq ($(INFO_FLAG),yes)
	$(info BACKEND      : $(BACKEND))
ifeq ($(BACKEND),cuda)
	$(info CUDA_PATH    : $(CUDA_PATH))
endif
ifeq ($(BACKEND),hip)
	$(info HIP_PATH     : $(HIP_PATH))
endif
	$(info COMPILER     : $(COMPILER))
	$(info GPU_ARCH     : $(GPU_ARCH))
	$(info GPU_MEM_MB   : $(GPU_MEM_MB))
	$(info )
endif

$(TARGET): $(TARGET).cu
	$(COMPILER) $< $(FLAGS) $(ARCH) -o $@ $(LIBS)

VERSION:
	$(COMPILER) --version 2>&1 | tee $(COMPILER)_version

run:
	./$(TARGET) $(MODE)

clean:
	rm -f *.o
	rm -f $(TARGET)
