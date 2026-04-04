
#!/bin/bash
#SBATCH --job-name=sz3_compress           # 作业名
#SBATCH --output=%j.out               # 标准输出日志，%j 是作业ID
#SBATCH --error=%j.err                # 标准错误日志
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH -p wholenode
#SBATCH -A cis240669 
export OMP_NUM_THREADS=1
module purge
module purge
module load gcc/11.2.0
module load openmpi/4.0.6
mpirun --bind-to core:overload-allowed -np $SLURM_NTASKS -mca btl ^openib  ./pio_nocmp 13 500 500 100
mpirun --bind-to core:overload-allowed -np $SLURM_NTASKS -mca btl ^openib  ./pio_sz3 xx 1E-3 13 500 500 100
mpirun --bind-to core:overload-allowed -np $SLURM_NTASKS -mca btl ^openib  ./pio_szo xx 1E-3 13 500 500 100
mpirun --bind-to core:overload-allowed -np $SLURM_NTASKS -mca btl ^openib  ./pio_zfp 13 500 500 100
