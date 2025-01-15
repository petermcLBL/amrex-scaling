To get the right modules on perlmutter:
```
module purge
module load cmake cudatoolkit PrgEnv-gnu
export LIBRARY_PATH=$CUDATOOLKIT_HOME/../../math_libs/lib64
export CPATH=$CUDATOOLKIT_HOME/../../math_libs/include
module load openmpi
module load python
```

Set AMREX_HOME to home directory of AMReX after installing it.

Set SPIRAL_HOME to home directory of
[SPIRAL](https://www.github.com/spiral-software/spiral-software)
after installing it.  Use the `develop` branch of SPIRAL and of associated
SPIRAL packages that are required for FFTX (see below).

Set FFTX_HOME to home directory of
[FFTX](https://www.github.com/spiral-software/fftx)
after installing it.  Use the `develop` branch.

Then to build this application:
```
make USE_FFTX=TRUE USE_CUDA=TRUE
```
