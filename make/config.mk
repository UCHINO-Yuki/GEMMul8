#===============
# User options
#===============

# Select GPU backend: cuda, hip, or auto.
BACKEND ?= auto

# Path to CUDA toolkit installation.
CUDA_PATH ?= /usr/local/cuda

# Path to HIP / ROCm toolkit installation.
HIP_PATH ?= /opt/rocm

# Target GPU architecture.
# Examples:
#   GPU_ARCH=90      # H100/H200/GH200
#   GPU_ARCH=gfx942  # MI300X
#   GPU_ARCH=auto    # auto-detect
GPU_ARCH ?= auto

# Temporary directory for compiler.
TEMPDIR ?= build/tmp
