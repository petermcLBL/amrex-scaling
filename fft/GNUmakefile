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
  DEFINES += -DUSE_FFTX -DFFTX_CUDA
  INCLUDE_LOCATIONS += $(FFTX_HOME)/include $(SPIRAL_HOME)/profiler/targets/include
  LIBRARY_LOCATIONS += $(FFTX_HOME)/lib $(CUDATOOLKIT_HOME)/lib64
  LIBRARIES += -Wl,--copy-dt-needed-entries -Wl,-rpath=$(FFTX_HOME)/lib -lfftx_mpi -lfftx_mddft_gpu -lfftx_imddft_gpu -lfftx_mdprdft_gpu -lfftx_imdprdft_gpu -lfftx_dftbat_gpu -lfftx_idftbat_gpu -lfftx_prdftbat_gpu -lfftx_iprdftbat_gpu -lfftx_rconv_gpu -lcudart
endif

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
