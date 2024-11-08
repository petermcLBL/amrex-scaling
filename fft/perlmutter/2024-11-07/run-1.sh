#!/bin/bash
#SBATCH --account=mp111_g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:10:00
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=debug

export MPICH_GPU_SUPPORT_ENABLED=1 
export SLURM_CPU_BIND="cores"

srun -n 1 ./main3d.gnu.MPI.CUDA.ex >& run-.25.ou

srun -n 4 ./main3d.gnu.MPI.CUDA.ex amrex.use_gpu_aware_mpi=1 \
                    n_cell_x=512 n_cell_y=512 n_cell_z=256 >& run-1.ou

