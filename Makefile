#===============
# GEMMul8 build
#===============

.DEFAULT_GOAL := all

include make/config.mk
include make/backend.mk
include make/functions.mk
include make/sources.mk
include make/rules.mk
include make/inst_common.mk

export TMPDIR := $(TEMPDIR)

ifeq ($(filter clean,$(MAKECMDGOALS)),)
INST_MK := $(sort $(shell find src -name instantiations.mk 2>/dev/null))
include $(INST_MK)
ALL_OBJ := $(CU_OBJ) $(INST_OBJ)
else
ALL_OBJ :=
endif

LINK_FLAGS :=

ifeq ($(shell uname -m),aarch64)
LINK_FLAGS += -Xlinker --stub-group-size=33554432
endif

.PHONY: all clean info compile_objects_banner

all: info $(STATIC_LIB) $(SHARED_LIB)

$(STATIC_LIB): $(ALL_OBJ)
	@mkdir -p lib
	@echo "Creating static library"
	@echo "AR  $@"
	@ar rcs $@ $^

$(SHARED_LIB): $(ALL_OBJ) $(HOOK_OBJ)
	@mkdir -p lib
	@echo "Creating shared library"
	@echo "LD  $@"
	@$(COMPILER) $(ARCH) -shared -o $@ $^ $(LIBS) $(LINK_FLAGS)

clean:
	rm -rf build lib $(COMPILE_INFO)
