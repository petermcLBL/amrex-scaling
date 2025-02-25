#!/bin/bash
### fill in with your account name
#SBATCH --account=
#SBATCH --time=0:10:00
#SBATCH --constraint=gpu&hbm40g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --qos=debug
#SBATCH --job-name=test_fft_c2c
#SBATCH --output=test_fft_c2c_256_perlmutter_out.%J
#SBATCH --error=test_fft_c2c_256_perlmutter_err.%J

export MPICH_GPU_SUPPORT_ENABLED=1 
export SLURM_CPU_BIND="cores"

date

echo "### 1 rank on 256^3"
srun -n 1 ../build/ffttest n_cell_x=256 n_cell_y=256 n_cell_z=256

echo "### 2 ranks on 256^3 use_gpu_aware_mpi=0"
srun -n 2 ../build/ffttest amrex.use_gpu_aware_mpi=0 n_cell_x=256 n_cell_y=256 n_cell_z=256

echo "### 2 ranks on 256^3 use_gpu_aware_mpi=1"
srun -n 2 ../build/ffttest amrex.use_gpu_aware_mpi=1 n_cell_x=256 n_cell_y=256 n_cell_z=256

echo "### 4 ranks on 256^3 use_gpu_aware_mpi=0"
srun -n 4 ../build/ffttest amrex.use_gpu_aware_mpi=0 n_cell_x=256 n_cell_y=256 n_cell_z=256

echo "### 4 ranks on 256^3 use_gpu_aware_mpi=1"
srun -n 4 ../build/ffttest amrex.use_gpu_aware_mpi=1 n_cell_x=256 n_cell_y=256 n_cell_z=256
