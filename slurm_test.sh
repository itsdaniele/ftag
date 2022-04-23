#!/bin/sh
#
# <https://doc.eresearch.unige.ch/hpc/slurm#gpgpu_jobs>

#SBATCH --partition=shared-gpu
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task 32
#SBATCH --nodes=1
#SBATCH --mem=24000
echo "I: full hostname: $(hostname -f)"

module load GCC/9.3.0

VERSION_CUDA='11.0.2'
module load CUDA/${VERSION_CUDA}

# if you need to know the allocated CUDA device, you can obtain it here:
echo "I: CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

echo "====="

source activate gecko
srun python test.py
