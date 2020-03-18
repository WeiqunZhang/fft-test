AMREX_HOME ?= ../amrex
FFTW_HOME ?= NOT_SET

DEBUG = FALSE
DIM = 3
COMP = gcc
TINY_PPROFILE = TRUE
USE_MPI = TRUE
USE_CUDA = FALSE
USE_FFTW = TRUE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs
include $(AMREX_HOME)/Src/Base/Make.package
include Make.package

ifneq ($(FFTW_HOME),NOT_SET)
  VPATH_LOCATIONS += $(FFTW_HOME)/include
  INCLUDE_LOCATIONS += $(FFTW_HOME)/include
  LIBRARY_LOCATIONS += $(FFTW_HOME)/lib
endif

ifeq ($(USE_FFTW),TRUE)
  DEFINES += -DUSE_FFTW
endif

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
