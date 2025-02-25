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
#SBATCH --job-name=FFTX_test_batch
#SBATCH --output=test_batch_perlmutter_out.%J
#SBATCH --error=test_batch_perlmutter_err.%J

export MPICH_GPU_SUPPORT_ENABLED=1 
export SLURM_CPU_BIND="cores"

date

echo "### 1 rank"
srun -n 1 ../build/ffttest

echo "### 2 ranks, use_gpu_aware_mpi=0"
srun -n 2 ../build/ffttest amrex.use_gpu_aware_mpi=0

echo "### 2 ranks, use_gpu_aware_mpi=1"
srun -n 2 ../build/ffttest amrex.use_gpu_aware_mpi=1

echo "### 4 ranks, use_gpu_aware_mpi=0"
srun -n 4 ../build/ffttest amrex.use_gpu_aware_mpi=0

echo "### 4 ranks, use_gpu_aware_mpi=1"
srun -n 4 ../build/ffttest amrex.use_gpu_aware_mpi=1
