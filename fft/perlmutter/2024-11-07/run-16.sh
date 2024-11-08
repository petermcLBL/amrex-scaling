#!/bin/bash
#SBATCH --account=mp111_g
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:10:00
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=regular

export MPICH_GPU_SUPPORT_ENABLED=1 
export SLURM_CPU_BIND="cores"

srun -n 64 ./main3d.gnu.MPI.CUDA.ex amrex.use_gpu_aware_mpi=1 \
                    n_cell_x=1024 n_cell_y=1024 n_cell_z=1024 >& run-16.ou

