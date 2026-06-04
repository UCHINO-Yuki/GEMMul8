#===============
# Explicit instantiations: src/gemm
#===============

GEMM_TYPE_TRIPLES_REAL := \
    s_s_s s_s_d s_d_s s_d_d \
    d_s_s d_s_d d_d_s d_d_d

GEMM_TYPE_TRIPLES_COMPLEX := \
    c_c_c c_c_z c_z_c c_z_z \
    z_c_c z_c_z z_z_c z_z_z

GEMMLT_BACKEND_CODES := i8 f8

GEMM_DIR := src/gemm
GEMM_SRC := $(GEMM_DIR)/cu_recipe/gemm.cu
GEMMLT_SRC := $(GEMM_DIR)/cu_recipe/gemmlt.cu

# gemm(): INT8 only.
define ADD_GEMM
$(eval $(call ADD_INST_OBJ,\
    $(GEMM_DIR)/gemm_i8_$(1)_$(2)_$(3),\
    $(GEMM_SRC),\
    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(1)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_BACKEND=INT8))
endef

$(foreach trip,$(GEMM_TYPE_TRIPLES_REAL),\
  $(call ADD_GEMM,$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip))))

$(foreach trip,$(GEMM_TYPE_TRIPLES_COMPLEX),\
  $(call ADD_GEMM,$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip))))

# gemmLt(): INT8 and FP8.
define ADD_GEMMLT
$(eval $(call ADD_INST_OBJ,\
    $(GEMM_DIR)/gemmlt_$(1)_$(2)_$(3)_$(4),\
    $(GEMMLT_SRC),\
    -DGEMMUL8_INST_TYPE_A=$(call type_cpp,$(2)) \
    -DGEMMUL8_INST_TYPE_B=$(call type_cpp,$(3)) \
    -DGEMMUL8_INST_TYPE_C=$(call type_cpp,$(4)) \
    -DGEMMUL8_INST_BACKEND=$(call backend_cpp,$(1))))
endef

$(foreach b,$(GEMMLT_BACKEND_CODES),\
  $(foreach trip,$(GEMM_TYPE_TRIPLES_REAL),\
    $(call ADD_GEMMLT,$(b),$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip)))))

$(foreach b,$(GEMMLT_BACKEND_CODES),\
  $(foreach trip,$(GEMM_TYPE_TRIPLES_COMPLEX),\
    $(call ADD_GEMMLT,$(b),$(call trip_T,$(trip)),$(call trip_A,$(trip)),$(call trip_B,$(trip)))))
