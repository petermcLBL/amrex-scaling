#!/bin/bash
#SBATCH --account=mp111_g
#SBATCH --nodes=256
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:10:00
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=regular

export MPICH_GPU_SUPPORT_ENABLED=1 
export SLURM_CPU_BIND="cores"

NNODES=256
NCORES=1024
NCELLS="n_cell_x=4096 n_cell_y=2048 n_cell_z=2048"

srun -n ${NCORES} ../../main3d.gnu.MPI.CUDA.ex amrex.use_gpu_aware_mpi=1 \
                    ${NCELLS} >& run-${NNODES}.ou
