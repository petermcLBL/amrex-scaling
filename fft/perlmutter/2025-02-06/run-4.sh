#!/bin/bash
#SBATCH --account=mp111_g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:10:00
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=debug

export MPICH_GPU_SUPPORT_ENABLED=1 
export SLURM_CPU_BIND="cores"

srun -n 16 ../../old-order.ex amrex.use_gpu_aware_mpi=1 \
                    n_cell_x=1024 n_cell_y=512 n_cell_z=512 >& run-4-oldorder.ou
srun -n 16 ../../new-order.ex amrex.use_gpu_aware_mpi=1 \
                    n_cell_x=1024 n_cell_y=512 n_cell_z=512 >& run-4-neworder.ou

export MPICH_OFI_NIC_POLICY=GPU

srun -n 16 ../../old-order.ex amrex.use_gpu_aware_mpi=1 \
                    n_cell_x=1024 n_cell_y=512 n_cell_z=512 >& run-4-oldorder-nci.ou
srun -n 16 ../../new-order.ex amrex.use_gpu_aware_mpi=1 \
                    n_cell_x=1024 n_cell_y=512 n_cell_z=512 >& run-4-neworder-nci.ou

