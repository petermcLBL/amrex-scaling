These scripts have been run on perlmutter:
* test_fft_128_perlmutter.sh
* test_fft_256_perlmutter.sh

First they were run with `FFTX_CUDA_AWARE_MPI` set to 0, and the outputs are in:
* test_fft_128_perlmutter_out.36243165
* test_fft_256_perlmutter_out.36243168

Then they were run with `FFTX_CUDA_AWARE_MPI` set to 1, and the outputs are in:
* test_fft_128_perlmutter_out.36284910
* test_fft_256_perlmutter_out.36284923
