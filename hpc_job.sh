#!/bin/bash
#SBATCH -J rnn_poetry
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o job_%j.log
#SBATCH -e job_%j.err
#SBATCH --time=02:00:00

echo "================================"
echo "  RNN Poetry Training on HPC"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# 进入项目目录（在 project 分区）
PROJ_DIR=~/project/CODE/rnn_poetry
cd $PROJ_DIR

# 检查数据
echo "Data: $(wc -l < data/poetry.txt) poems"
echo "Python: $(which python3)"
echo "Pytorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# 运行训练
echo "Training started: $(date)"
python3 main.py --train

echo "Training done: $(date)"

# 测试生成
echo "=== Generation Test ==="
python3 -c "
from main import generate, load_vocab, PoetryLSTM
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab = load_vocab()
model = PoetryLSTM(vocab['size']).to(device)
ckpt = torch.load('checkpoints/lstm_poetry_best.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
for start in ['春江', '明月', '白日', '青山', '秋水', '花落', '夜雨']:
    r = generate(start, model, vocab, max_len=24, temperature=0.8, top_k=8)
    print(f'{start} → {r}')
print('--- Acrostic ---')
for name in ['李白', '王维', '杜甫']:
    from main import gen_acrostic
    r = gen_acrostic(name, model, vocab)
    print(f'{name}: {r}')
"
