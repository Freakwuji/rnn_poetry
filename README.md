# RNN 古诗词生成

基于 PyTorch LSTM 的中国古诗词生成系统，支持唐诗、宋词、元曲、明清诗等多朝代风格生成。已部署至武汉大学超算中心（HPC）完成 GPU 加速训练。

---

## 项目结构

```
rnn_poetry/
├── main.py              # 主入口：训练/预测/生成
├── model.py             # LSTM 模型定义
├── config.py            # 超参数配置
├── process.py           # 数据加载与词表构建
├── utils.py             # 工具函数
├── hpc_job.sh           # 超算 Slurm 作业脚本
├── deploy_to_hpc.bat    # Windows 部署到超算脚本
├── deploy_to_hpc.ps1    # PowerShell 部署脚本
├── requirements.txt     # 依赖清单
├── README.md            # 本文件
├── data/
│   ├── poetry.txt       # 训练语料（138,629 首）
│   ├── vocab.json       # 词表（6,487 字）
│   └── *.py             # 数据扩充脚本
├── checkpoints/
│   └── lstm_poetry_best.pt  # 最佳模型权重
└── __pycache__/
```

---

## 数据集

| 朝代 | 数量 | 占比 |
|:----|:----:|:----:|
| 唐诗 | 57,598 首 | 41.5% |
| 宋诗 | 60,000 首 | 43.3% |
| 宋词 | 8,000 首 | 5.8% |
| 元曲 | 5,000 首 | 3.6% |
| 明清诗 | 8,000 首 | 5.8% |
| **合计** | **138,629 首** | **100%** |

---

## 模型架构

```
Embedding(6487→256) → LSTM(256→512, 3层) → Dropout(0.5) → Linear(512→6487)
```

| 参数 | 值 |
|:----|:----|
| 词嵌入维度 | 256 |
| 隐藏层维度 | 512 |
| LSTM 层数 | 3 |
| Dropout | 0.5 |
| 总参数量 | ~3.8M |
| 词表大小 | 6,487 字 |

---

## 环境要求

```bash
pip install -r requirements.txt
# 依赖：torch, numpy
```

---

## 用法

### 训练

```bash
# 本地训练（CPU）
python main.py --train

# 修改训练轮数
# 编辑 config.py 中的 EPOCHS = 200
```

### 生成古诗

```bash
# 默认生成示例
python main.py --predict

# 指定起始字生成
python main.py --start "春江"

# 藏头诗
python main.py --acrostic "黎坤朗"
```

### 代码中调用

```python
from main import generate, gen_acrostic, load_vocab, PoetryLSTM
import torch

vocab = load_vocab()
model = PoetryLSTM(vocab['size']).to('cpu')
ckpt = torch.load('checkpoints/lstm_poetry_best.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# 生成
result = generate('春江', model, vocab, max_len=24, temperature=0.8, top_k=8)
print(result)

# 藏头诗
result = gen_acrostic('李白', model, vocab)
print(result)
```

---

## 参数调优

| 参数 | 说明 | 推荐值 |
|:----|:----|:------:|
| temperature | 生成温度（越高越随机） | 0.6~0.9 |
| top_k | 候选字数量 | 5~10 |
| max_len | 最大生成长度 | 24~64 |

**温度参考：**
- 0.6：保守，更工整
- 0.8：平衡，词汇最丰富（推荐）
- 1.0：激进，更多变化

---

## 超算（HPC）部署

项目已在武汉大学超算中心完成训练。

### 准备工作

```bash
# 1. 上传代码到超算
cd rnn_poetry
tar -czf rnn_poetry.tar.gz .
scp rnn_poetry.tar.gz clli_04@swarm01.whu.edu.cn:~/

# 2. 登录超算解压
ssh clli_04@swarm01.whu.edu.cn
tar xzf rnn_poetry.tar.gz -C ~/project/CODE/rnn_poetry

# 3. 安装依赖
pip install torch numpy
```

### 提交作业

```bash
cd ~/project/CODE/rnn_poetry
sbatch hpc_job.sh
```

### 查看状态

```bash
squeue -u clli_04          # 查看队列
tail -f job_*.log           # 实时看训练输出
squeue -j <job_id>          # 查看特定作业
```

---

## 训练结果

| 试验 | 数据 | 轮数 | 模型 | 准确率 |
|:----|:----|:----:|:----:|:-----:|
| 纯唐诗 | 57,598 首 | 200 | 256d-3L | 6.07% |
| 多朝代 | 138,629 首 | 200 | 256d-3L | **17.72%** |

---

## 生成示例

### 唐诗风格

```
春江 → 春江水月上寒花满月深风起水烟波起月斜来月夜秋云上水空
明月 → 明月前秋水满花飞落白云春色不堪见处心心去有时心不得身
青山 → 青山流在旧山下一襟烟雨六朝风月秋云一蓑梅花中小径竹一
```

### 藏头诗

```python
gen_acrostic('李白', model, vocab)
# → 李无人不可求我，白云间水水清沈
```

---

## 数据扩充

如需扩充训练语料，可运行 `data/` 下的脚本：

```bash
cd data
python add_dynasties.py    # 添加宋元明清诗词
python expand_to_100k.py   # 扩充至 10 万首
```

---

## 参考

- 原始唐诗数据：[taishan1994/pytorch_peot_rnn](https://github.com/taishan1994/pytorch_peot_rnn)
- 武汉大学超算中心文档：[docs.hpc.whu.edu.cn](https://docs.hpc.whu.edu.cn/)
