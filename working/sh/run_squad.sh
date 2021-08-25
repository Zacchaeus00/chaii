#!/bin/bash
#SBATCH --job-name=run_squad           # 任务名
#SBATCH --nodes=1                   # 这里不用动 多节点脚本请查官方文档
#SBATCH --ntasks=1                  # 这里不用动 多任务脚本请查官方文档
#SBATCH --cpus-per-task=8           # 要几块CPU (一般4块就够用了)
#SBATCH --mem=16GB                  # 最大内存
#SBATCH --time=24:00:00             # 运行时间上限
#SBATCH --mail-type=END             # ALL / END
#SBATCH --mail-user=yw3642@nyu.edu  # 结束之后给哪里发邮件
#SBATCH --output=%x%A.out         # 正常输出写入的文件
#SBATCH --error=%x%A.err          # 报错信息写入的文件
#SBATCH --gres=gpu:2                # 需要几块GPU (同时最多8块)
#SBATCH -p aquila                   # 有GPU的partition
#SBATCH --nodelist=agpu7            # 3090

module purge                        # 清除所有已加载的模块
module load anaconda3 cuda/11.1.1              # 加载anaconda (load virtual env for training)

nvidia-smi
nvcc --version
cd /gpfsnyu/scratch/yw3642/chaii/working/src     # 切到程序目录

echo "START"               # 输出起始信息
source deactivate
source /gpfsnyu/packages/anaconda3/5.2.0/bin/activate kaggle          # 调用 virtual env
export SQUAD_DIR=/gpfsnyu/scratch/yw3642/chaii/input/squad2
python -u run_squad.py \
  --model_type rembert \
  --model_name_or_path ../../input/google-rembert \
  --do_train \
  --do_eval \
  --train_file $SQUAD_DIR/train-v2.0.json \
  --predict_file $SQUAD_DIR/dev-v2.0.json \
  --per_gpu_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --num_train_epochs 3 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir model/debug_run_squad/
echo "FINISH"                       # 输出起始信息