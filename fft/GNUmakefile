AMREX_HOME ?= ../..

DEBUG	= FALSE

DIM	= 3

COMP    = gcc

USE_MPI   = TRUE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_SYCL  = FALSE

USE_FFT = TRUE
USE_HEFFTE = FALSE
USE_FFTX = FALSE

BL_NO_FORT = TRUE

TINY_PROFILE = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

ifeq ($(USE_HEFFTE),TRUE)
  DEFINES += -DUSE_HEFFTE
  INCLUDE_LOCATIONS += $(HEFFTE_HOME)/include
  LIBRARY_LOCATIONS += $(HEFFTE_HOME)/lib
  LIBRARIES += -lheffte
endif

ifeq ($(USE_FFTX),TRUE)
  DEFINES += -DUSE_FFTX
  INCLUDE_LOCATIONS += $(FFTX_HOME)/include $(SPIRAL_HOME)/profiler/targets/include
  LIBRARY_LOCATIONS += $(FFTX_HOME)/lib
  LIBRARIES += -Wl,--rpath=$(FFTX_HOME)/lib -lfftx_mpi
endif

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
