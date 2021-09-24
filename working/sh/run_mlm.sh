#!/bin/bash
#SBATCH --job-name=run_mlm           # 任务名
#SBATCH --nodes=1                   # 这里不用动 多节点脚本请查官方文档
#SBATCH --ntasks=1                  # 这里不用动 多任务脚本请查官方文档
#SBATCH --cpus-per-task=8           # 要几块CPU (一般4块就够用了)
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB                  # 最大内存
#SBATCH --mail-type=END             # ALL / END
#SBATCH --mail-user=yw3642@nyu.edu  # 结束之后给哪里发邮件
#SBATCH --output=%x%A.out         # 正常输出写入的文件
#SBATCH --error=%x%A.err          # 报错信息写入的文件
#SBATCH --gres=gpu:1                # 需要几块GPU (同时最多8块)
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
python run_mlm.py \
--seed 42 \
--model_name_or_path ../../input/xlm-roberta-large \
--train_file ../../input/wiki/hita-wiki-0920.csv \
--per_device_train_batch_size 8 \
--fp16 \
--output_dir ../model/0924pretrain-test \
--warmup_ratio 0.1 \
--num_train_epochs 3 \
--mlm_probability 0.15 \
--learning_rate 3e-5 \
--gradient_accumulation_steps 1 \
--validation_split_percentage 5 \
--load_best_model_at_end \
--evaluation_strategy 'epoch' \
--save_strategy 'epoch' \
--report_to 'none'
echo "FINISH"                       # 输出起始信息