* build amrex in amrex directory
  $ mkdir build
  $ cd build
  $ cmake .. -DAMReX_GPU_BACKEND=CUDA -DAMReX_FFT=ON
  $ make -j8
  $ make install
