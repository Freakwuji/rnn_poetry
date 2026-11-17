# RNN 古诗词生成

基于 PyTorch 的中国古诗词生成系统，支持多种 RNN + Attention 模型对比实验。

---

## 一、项目简介

### 1.1 任务目标

给定一个字，预测下一个字（语言模型），并生成完整的古诗。

```
输入: "床前明"
模型学习: 床→前, 前→明, 明→月, 月→光
输出: "床前明月光"
```

### 1.2 改进内容

| 模块 | 原 Demo | 改进后 |
|------|---------|--------|
| 数据 | 仅唐诗57k | chinese-poetry数据集（唐诗+宋诗） |
| 模型 | 基础LSTM | 7种模型对比 |
| 训练 | StepLR | Warmup+CosineAnnealing+早停+AMP |
| 评价 | Perplexity | 多维度：规则化+BLEU+人工+AI |

---

## 二、数据集

### 2.1 数据来源

使用 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) 数据集（GitHub上最全的中文古诗词数据集）。

### 2.2 数据构成

| 朝代 | 数量 | 说明 |
|------|------|------|
| 唐诗 | ~55,000首 | 全唐诗 |
| 宋诗 | ~250,000首 | 全宋诗 |
| **合计** | **~300,000首** | 仅保留五言和七言 |

### 2.3 为什么只保留诗，不保留词和曲？

| 类型 | 结构 | 是否保留 |
|------|------|----------|
| 古诗 | 五言/七言，固定字数 | ✓ 保留 |
| 宋词 | 长短句，字数不固定 | ✗ 不保留 |
| 元曲 | 长短句，更口语化 | ✗ 不保留 |

**原因**：
- 古诗生成需要输出规整的五言/七言
- 词和曲是长短句，会干扰模型学习
- 只保留诗，生成质量更好

### 2.4 数据清洗

```python
# 清洗规则
1. 只保留五言和七言诗
2. 去重（内容完全相同的诗）
3. 过滤乱码（非中文字符超过10%）
4. 统一标点符号
```

---

## 三、模型结构

### 3.1 7个模型对比

| # | 模型 | 结构 | 参数量 | 对比目的 |
|---|------|------|--------|----------|
| 1 | LSTM | Embedding→LSTM→Linear | ~7.5M | 基线1 |
| 2 | GRU | Embedding→GRU→Linear | ~6.6M | 基线2 |
| 3 | LSTM+Attn | Embedding→LSTM→Bahdanau→Linear | ~8.6M | 验证Attention |
| 4 | GRU+Attn | Embedding→GRU→Bahdanau→Linear | ~7.7M | 对比 |
| 5 | BiLSTM+Attn | Embedding→BiLSTM→Bahdanau→Linear | ~16.2M | 验证双向 |
| 6 | LSTM+MultiAttn | Embedding→LSTM→MultiHead→Linear | ~8.8M | 验证多头 |
| 7 | BiLSTM+MultiAttn | Embedding→BiLSTM→MultiHead→Linear | ~18.3M | 完整版 |

### 3.2 对比逻辑

```
1 vs 2: LSTM vs GRU → 谁更好
1 vs 3: 验证Attention效果
3 vs 4: LSTM+Attn vs GRU+Attn → 谁更适合加Attention
3 vs 5: 验证双向效果
3 vs 6: 验证多头效果
1 vs 7: 完整改进效果
```

### 3.3 结构图

```
┌─────────────────────────────────────────────────────────────┐
│              LSTM + Bahdanau Attention (模型3)               │
├─────────────────────────────────────────────────────────────┤
│  输入: [床, 前, 明, 月] (字索引)                               │
│      ↓                                                      │
│  Embedding层: 把每个字变成256维向量                              │
│      ↓                                                      │
│  LSTM层(2层): 提取时序特征，输出512维                            │
│      ↓                                                      │
│  Bahdanau Attention: 关注前面的关键位置                         │
│      ↓                                                      │
│  Dropout(0.5): 防止过拟合                                    │
│      ↓                                                      │
│  Linear层: 映射到词表大小(6000+)                               │
│  输出: 每个字的概率分布                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、训练优化

### 4.1 学习率调度

```
Epoch 1-5:   0.0002 → 0.001 (Warmup，线性增加)
Epoch 6-200: 0.001 → 0.00001 (Cosine，余弦衰减)
```

### 4.2 早停机制

当验证集loss连续15轮不下降时，提前停止训练。

### 4.3 混合精度训练 (AMP)

- 显存减半
- 速度提升1.5-2倍

### 4.4 生成策略

| 参数 | 默认值 | 说明 |
|------|--------|------|
| temperature | 0.8 | 温度采样 |
| top_k | 8 | Top-K采样 |
| top_p | 0.9 | Nucleus采样 |
| repetition_penalty | 1.2 | 重复惩罚 |

---

## 五、评价系统

### 5.1 多角度评价

| 评价方式 | 指标 | 说明 |
|----------|------|------|
| 自动指标 | Perplexity, BLEU | 客观指标 |
| 规则化评价 | 押韵、对仗、重复度、字数、意象 | 结构评价 |
| 人工评价 | 通顺度、意境、押韵、创意 | 主观评价 |
| AI评价 | 多维度打分 | 综合评价 |

### 5.2 规则化评价维度

| 维度 | 权重 | 检测方法 |
|------|------|----------|
| 押韵 | 25% | 检查句尾韵母 |
| 对仗 | 20% | 检查字数整齐度 |
| 重复度 | 20% | 计算重复率 |
| 字数整齐 | 15% | 是否五言/七言 |
| 意象丰富 | 20% | 检查意象词汇 |

---

## 六、快速开始

### 6.1 安装依赖

```bash
pip install torch numpy matplotlib
```

### 6.2 下载数据

```bash
python data/download_dataset.py
```

### 6.3 训练模型

```bash
# 训练单个模型
python main.py --train --model lstm_attn --epochs 200

# 对比实验（7个模型）
python main.py --compare --epochs 200
```

### 6.4 生成古诗

```bash
python main.py --predict --model lstm_attn
```

### 6.5 生成可视化

```bash
python visualize.py
```

---

## 七、超算运行

### 7.1 上传代码

```powershell
cd "d:\Desktop\AI实训\3古诗生成实验\rnn_poetry"
tar -czf rnn_poetry.tar.gz . --exclude=__pycache__
scp rnn_poetry.tar.gz your_hpc_user@swarm01.whu.edu.cn:~/
```

### 7.2 提交作业

```bash
ssh your_hpc_user@swarm01.whu.edu.cn
mkdir -p ~/project/CODE/rnn_poetry && cd ~/project/CODE/rnn_poetry
tar xzf ~/rnn_poetry.tar.gz
sbatch run_experiments.sh
```

### 7.3 查看进度

```bash
squeue -u your_hpc_user
tail -f job_*.log
```

---

## 八、项目结构

```
rnn_poetry/
├── main.py                 # 主程序
├── model.py                # 7种模型
├── attention.py            # Attention机制
├── trainer.py              # 训练器
├── evaluate.py             # 评价系统
├── config.py               # 配置文件
├── process.py              # 数据处理
├── visualize.py            # 可视化模块
├── run_experiments.sh      # 超算实验脚本
├── requirements.txt        # 依赖清单
├── 代码讲解.md              # 代码详细讲解
├── 完整最终版操作方案.md      # 操作指南
├── 优化改进总结报告.md        # 改进报告
├── 使用指南.md              # 超算使用指南
├── data/
│   ├── download_dataset.py # 数据下载脚本
│   ├── poetry.txt          # 训练语料
│   └── vocab.json          # 词表
├── checkpoints/            # 模型权重
└── results/                # 实验结果
```

---

## 九、命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | lstm_attn | 模型类型 |
| `--epochs` | 200 | 训练轮数 |
| `--batch_size` | 128 | 批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--hidden_dim` | 512 | 隐藏层维度 |
| `--temperature` | 0.8 | 生成温度 |
| `--top_k` | 8 | Top-K采样 |
| `--top_p` | 0.9 | Top-P采样 |
| `--use_amp` | - | 启用混合精度 |

---

## 十、参考

- 原始代码：[taishan1994/pytorch_peot_rnn](https://github.com/taishan1994/pytorch_peot_rnn)
- 数据集：[chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
- 武汉大学超算中心：[docs.hpc.whu.edu.cn](https://docs.hpc.whu.edu.cn/)
