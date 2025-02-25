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

srun -n 1 ../../old-order.ex >& run-.25-oldorder.ou
srun -n 1 ../../new-order.ex >& run-.25-neworder.ou

srun -n 4 ../../old-order.ex amrex.use_gpu_aware_mpi=1 \
                    n_cell_x=512 n_cell_y=512 n_cell_z=256 >& run-1-oldorder.ou
srun -n 4 ../../new-order.ex amrex.use_gpu_aware_mpi=1 \
                    n_cell_x=512 n_cell_y=512 n_cell_z=256 >& run-1-neworder.ou
