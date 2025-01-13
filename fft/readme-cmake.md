* build amrex in amrex directory
  $ mkdir build
  $ cd build
  $ cmake .. -DAMReX_GPU_BACKEND=CUDA -DAMReX_FFT=ON
  $ make -j8
  $ make install

* build ffttest in amrex-scaling/fft
  $ mkdir build
  $ cd build
  $ cmake .. -DAMReX_ROOT=${AMREX_HOME}/installdir -DFFTX_HOME=${FFTX_HOME} -DSPIRAL_HOME=${SPIRAL_HOME} -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
  $ make -j8
