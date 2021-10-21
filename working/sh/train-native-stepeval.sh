#!/bin/bash
#SBATCH --job-name=stepeval         # 任务名
#SBATCH --nodes=1                   # 这里不用动 多节点脚本请查官方文档
#SBATCH --ntasks=1                  # 这里不用动 多任务脚本请查官方文档
#SBATCH --cpus-per-task=8           # 要几块CPU (一般4块就够用了)
#SBATCH --mem=16GB                  # 最大内存
#SBATCH --time=24:00:00             # 运行时间上限
#SBATCH --mail-type=END             # ALL / END
#SBATCH --mail-user=yw3642@nyu.edu  # 结束之后给哪里发邮件
#SBATCH --output=%x%A.out         # 正常输出写入的文件
#SBATCH --error=%x%A.err          # 报错信息写入的文件
#SBATCH --gres=gpu:1                # 需要几块GPU (同时最多8块)
#SBATCH -p aquila                   # 有GPU的partition
#SBATCH --nodelist=agpu7            # apu7: 3090, apu5/6: 2080ti, gpu6: p100

module purge                        # 清除所有已加载的模块
module load anaconda3 cuda/11.1.1              # 加载anaconda (load virtual env for training)

nvidia-smi
nvcc --version
cd /gpfsnyu/scratch/yw3642/chaii/working/src     # 切到程序目录

echo "START"               # 输出起始信息
source deactivate
source /gpfsnyu/packages/anaconda3/5.2.0/bin/activate kaggle          # 调用 virtual env
python -u train-native-stepeval.py \
--model_checkpoint /gpfsnyu/scratch/yw3642/chaii/working/model/2021-10-20-17:05:05/checkpoint-97887 \
--train_path /gpfsnyu/scratch/yw3642/chaii/input/train0917/merged0917.csv \
--max_length 512 \
--doc_stride 128 \
--epochs 3 \
--batch_size 4 \
--accumulation_steps 1 \
--lr 1e-5 \
--optimizer adamw \
--weight_decay 0 \
--scheduler cosann \
--warmup_ratio 0.1 \
--dropout 0.1 \
--eval_steps 1000 \
--metric nonzero_jaccard_per \
--downext \
--seed 3407
echo "FINISH"                       # 输出起始信息