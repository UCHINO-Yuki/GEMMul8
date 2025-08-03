#===============
# path to cuda
#===============
CUDA_PATH := /usr/local/cuda

#===============
# "ozIMMU_EF := yes" if you use the ozIMMU_EF (accelerated Ozaki scheme 1) in a sample code.
#===============
ozIMMU_EF := yes

#===============
# path to ozIMMU_EF (Set this if "ozIMMU_EF := yes")
#===============
ozIMMU_EF_DIR := ../../ozIMMU_EF

#===============
# "cuMpSGEMM := yes" if you use the cuMpSGEMM in a sample code and compute_cap < 100.
#===============
cuMpSGEMM := yes

#===============
# path to cuMpSGEMM (Set this if "cuMpSGEMM := yes")
#===============
cuMpSGEMM_DIR := ../../cuMpSGEMM


export PATH := $(CUDA_PATH)/bin:$(PATH)
export PATH := $(CUDA_PATH)/bin/lib64:$(PATH)

GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
ARCHS := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)
GPU_MEM_MB := $(shell nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
NVCCFLAGS := -std=c++20 $(ARCHS) -lnvidia-ml -I../include -O3 -DGPU_MEM_MB=$(GPU_MEM_MB)

LIBS := -lcublas -lcudart -lcuda -L../lib -lgemmul8
LD:=

ifneq ($(filter $(GPU_ARCH), 80 86 87 89 90),)
else
    cuMpSGEMM := no
endif

ifeq ($(ozIMMU_EF),yes)
NVCCFLAGS += -DozIMMU_EF_FLAG -I$(ozIMMU_EF_DIR)/include
LIBS += -L$(ozIMMU_EF_DIR)/build -lozimmu
ifeq ($(cuMpSGEMM),yes)
NVCCFLAGS += -DcuMpSGEMM_FLAG -I$(cuMpSGEMM_DIR)/include
LIBS += -L$(cuMpSGEMM_DIR)/build -lcumpsgemm
LD+=$(ozIMMU_EF_DIR)/build/libozimmu.so:$(cuMpSGEMM_DIR)/build/libcumpsgemm.so
else
LD+=$(ozIMMU_EF_DIR)/build/libozimmu.so
endif
else
ifeq ($(cuMpSGEMM),yes)
NVCCFLAGS += -DcuMpSGEMM_FLAG -I$(cuMpSGEMM_DIR)/include
LIBS += -L$(cuMpSGEMM_DIR)/build -lcumpsgemm
LD+=$(cuMpSGEMM_DIR)/build/libcumpsgemm.so
endif
endif

$(info CUDA_PATH    : $(CUDA_PATH))
$(info GPU_ARCH     : $(GPU_ARCH))
$(info ozIMMU_EF    : $(ozIMMU_EF))
$(info ozIMMU_EF_DIR: $(ozIMMU_EF_DIR))
$(info cuMpSGEMM    : $(cuMpSGEMM))
$(info cuMpSGEMM_DIR: $(cuMpSGEMM_DIR))

TARGET_d := test_double
TARGET_f := test_float

all: $(TARGET_d) $(TARGET_f) VERSION

$(TARGET_d): $(TARGET_d).cu
	nvcc $< $(NVCCFLAGS) $(LIBS) -o $@

$(TARGET_f): $(TARGET_f).cu
	nvcc $< $(NVCCFLAGS) $(LIBS) -o $@

VERSION:
	nvcc --version 2>&1 | tee nvcc_version

test_f:
	LD_LIBRARY_PATH=$(CUDA_PATH)/lib64 LD_PRELOAD=$(LD) ./$(TARGET_f) $(MODE)

test_d:
	LD_LIBRARY_PATH=$(CUDA_PATH)/lib64 LD_PRELOAD=$(LD) ./$(TARGET_d) $(MODE)

clean:
	rm -f *.o
	rm -f $(TARGET_d) $(TARGET_f)

