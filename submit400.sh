#!/bin/bash 

#SBATCH --job-name=ly_dinoV3_runtest400            # 作业名
#SBATCH --comment="400 data for each kind of disease."    # 作业描述

#SBATCH --partition=L40      # 使用哪个分区
#SBATCH --gres=gpu:l40:1

#SBATCH --output=%x_%j.out       # 输出文件
#SBATCH --error=%x_%j.err        # 错误输出文件

#SBATCH --mail-type=end
#SBATCH --mail-user=2452443@tongji.edu.cn

#SBATCH --nodes=1               
#SBATCH --ntasks=1               
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

module load cuda/11.8
echo ${SLURM_JOB_NODELIST}
echo start on $(date)              

# 编写或调用你自己的程序
echo data prepared.

python /share/home/u23171/longyi/dinoV3/src/train_detector.py

echo end on $(date)

