These scripts have been run on perlmutter:
* test_fft_c2c_128_perlmutter.sh
* test_fft_c2c_256_perlmutter.sh

First they were run with `FFTX_CUDA_AWARE_MPI` set to 0, and the outputs are in:
* test_fft_c2c_128_perlmutter_out.36243335
* test_fft_c2c_256_perlmutter_out.36243338

Then they were run with `FFTX_CUDA_AWARE_MPI` set to 1, and the outputs are in:
* test_fft_c2c_128_perlmutter_out.36284949
* test_fft_c2c_256_perlmutter_out.36284952
