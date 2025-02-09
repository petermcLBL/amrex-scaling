#!/bin/bash
#SBATCH --account=mp111_g
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:10:00
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=regular

export MPICH_GPU_SUPPORT_ENABLED=1 
export SLURM_CPU_BIND="cores"

NNODES=64
NCORES=256
NCELLS="n_cell_x=512 n_cell_y=256 n_cell_z=256"

srun -n ${NCORES} ../../main3d.gnu.MPI.CUDA.ex amrex.use_gpu_aware_mpi=1 \
                    ${NCELLS} batch_size=1 >& run-${NNODES}-b1.ou

srun -n ${NCORES} ../../main3d.gnu.MPI.CUDA.ex amrex.use_gpu_aware_mpi=1 \
                    ${NCELLS} batch_size=10 >& run-${NNODES}-b10.ou

srun -n ${NCORES} ../../main3d.gnu.MPI.CUDA.ex amrex.use_gpu_aware_mpi=1 \
                    ${NCELLS} batch_size=100 >& run-${NNODES}-b100.ou

srun -n ${NCORES} ../../main3d.gnu.MPI.CUDA.ex amrex.use_gpu_aware_mpi=1 \
                    ${NCELLS} batch_size=1000 >& run-${NNODES}-b1000.ou
