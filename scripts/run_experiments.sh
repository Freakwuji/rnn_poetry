#!/bin/bash
#SBATCH -J rnn_poetry
#SBATCH -p hpxg
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o job_%j.log
#SBATCH -e job_%j.err
#SBATCH --time=24:00:00

echo "================================"
echo "  RNN Poetry - 7模型对比实验"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# 进入项目目录
PROJ_DIR=~/project/CODE/rnn_poetry
cd $PROJ_DIR

# 检查环境
echo ""
echo "[环境检查]"
echo "Python: $(which python3)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# 下载新数据集（chinese-poetry，30万+首）
echo ""
echo "[数据] 下载 chinese-poetry 数据集..."
python3 data/download_dataset.py
echo "数据: $(wc -l < data/poetry.txt) 行"

# 创建结果目录
mkdir -p results

# 设置环境变量
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================
# 并行训练函数
# ============================================================

train_model() {
    local model_name=$1
    local epochs=${2:-200}

    echo "[训练] 开始训练 $model_name (CPU)"
    python3 main.py --train \
        --model $model_name \
        --epochs $epochs \
        --batch_size 64 \
        --lr 0.001 \
        --hidden_dim 512 \
        --num_layers 2

    echo "[完成] $model_name 训练完成"
}

# ============================================================
# 第一轮：训练 LSTM 和 GRU
# ============================================================
echo ""
echo "================================"
echo "  第一轮: LSTM 和 GRU"
echo "================================"
echo "开始: $(date)"

train_model lstm 200
train_model gru 200

echo "第一轮完成: $(date)"

# ============================================================
# 第二轮：训练 LSTM+Attn 和 GRU+Attn
# ============================================================
echo ""
echo "================================"
echo "  第二轮: LSTM+Attn 和 GRU+Attn"
echo "================================"
echo "开始: $(date)"

train_model lstm_attn 200
train_model gru_attn 200

echo "第二轮完成: $(date)"

# ============================================================
# 第三轮：训练 BiLSTM+Attn 和 LSTM+MultiAttn
# ============================================================
echo ""
echo "================================"
echo "  第三轮: BiLSTM+Attn 和 LSTM+MultiAttn"
echo "================================"
echo "开始: $(date)"

train_model bilstm_attn 200
train_model lstm_multi_attn 200

echo "第三轮完成: $(date)"

# ============================================================
# 第四轮：训练 BiLSTM+MultiAttn（完整版）
# ============================================================
echo ""
echo "================================"
echo "  第四轮: BiLSTM+MultiAttn（完整版）"
echo "================================"
echo "开始: $(date)"

train_model bilstm_multi_attn 200

echo "第四轮完成: $(date)"

# ============================================================
# 生成示例并评价
# ============================================================
echo ""
echo "================================"
echo "  生成示例和评价"
echo "================================"
echo "开始: $(date)"

python3 -c "
import sys
sys.path.insert(0, '.')
import torch
import json
from main import generate_poems
from model import get_model
from process import load_vocab
from evaluate import RuleBasedEvaluator
import config

# 加载词表
vocab = load_vocab()

# 模型列表（7个模型）
models = {
    'LSTM': 'lstm',
    'GRU': 'gru',
    'LSTM+Attn': 'lstm_attn',
    'GRU+Attn': 'gru_attn',
    'BiLSTM+Attn': 'bilstm_attn',
    'LSTM+MultiAttn': 'lstm_multi_attn',
    'BiLSTM+MultiAttn': 'bilstm_multi_attn',
}

start_words = ['春江', '明月', '青山', '花落', '夜雨', '秋水', '白日']
evaluator = RuleBasedEvaluator()
all_results = []

for name, model_type in models.items():
    print(f'\n--- {name} ---')
    model = get_model(model_type, vocab['size'])
    model = model.to(config.DEVICE)

    # 加载权重
    try:
        ckpt = torch.load(f'checkpoints/{model_type}_best.pt', map_location=config.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'  加载模型: checkpoints/{model_type}_best.pt')
    except:
        print(f'  模型不存在: {model_type}')
        continue

    # 生成
    poems = generate_poems(model, vocab, start_words, max_len=24, temperature=0.8, top_k=8)

    # 评价
    scores = []
    for start, poem in poems:
        score, details = evaluator.evaluate_poem(poem)
        scores.append(score)
        print(f'  {start}: {poem[:20]}... ({score:.3f})')

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f'  平均分: {avg_score:.3f}')

    all_results.append({
        'model': name,
        'scores': scores,
        'avg_score': avg_score,
        'poems': [(s, p) for s, p in poems]
    })

# 保存结果
with open('results/final_evaluation.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f'\n结果已保存: results/final_evaluation.json')

# 打印对比表格
print('\n================================')
print('  最终对比')
print('================================')
print('| 模型 | 平均分 | 最高分 | 最低分 |')
print('|------|--------|--------|--------|')
for r in all_results:
    print(f\"| {r['model']:18s} | {r['avg_score']:.3f} | {max(r['scores']):.3f} | {min(r['scores']):.3f} |\")
"

echo "完成: $(date)"

# ============================================================
# 生成可视化图表
# ============================================================
echo ""
echo "================================"
echo "  生成可视化图表"
echo "================================"
echo "开始: $(date)"

python3 -c "
import sys
sys.path.insert(0, '.')
from visualize import plot_all
plot_all('results')
"

echo "完成: $(date)"

# ============================================================
# 总结
# ============================================================
echo ""
echo "================================"
echo "  实验完成!"
echo "================================"
echo "结束: $(date)"
echo ""
echo "结果文件:"
ls -lh results/
echo ""
echo "模型文件:"
ls -lh checkpoints/
echo ""
echo "可视化图表:"
ls -lh results/*.png 2>/dev/null || echo "  无图表文件"
