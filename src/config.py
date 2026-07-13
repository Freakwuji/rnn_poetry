"""
配置文件 — RNN 古诗词生成
"""
import os

# ── 路径 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ── 数据 ──
DATA_PATH = os.path.join(DATA_DIR, 'poetry.txt')   # 原始语料
VOCAB_PATH = os.path.join(DATA_DIR, 'vocab.json')   # 词表缓存

# ── 模型 ──
MODEL_NAME = 'lstm_poetry'           # 模型名称
EMBEDDING_DIM = 128                   # 词嵌入维度
HIDDEN_DIM = 256                      # 隐藏层维度
NUM_LAYERS = 2                        # LSTM 层数
DROPOUT = 0.5                         # Dropout 概率

# ── 训练 ──
BATCH_SIZE = 128                      # 批次大小
SEQ_LENGTH = 48                       # 序列长度
EPOCHS = 30                           # 训练轮数
LEARNING_RATE = 0.001                 # 学习率
NUM_WORKERS = 2                       # 数据加载线程（HPC多核环境）

# ── 生成 ──
MAX_GEN_LENGTH = 64                   # 最大生成长度
TEMPERATURE = 0.8                     # 生成温度（越高越随机）
TOP_K = 8                             # 采样时只从前K个候选字选

# ── 设备 ──
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
