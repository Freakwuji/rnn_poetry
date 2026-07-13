"""
RNN 模型  基于 LSTM 的古诗词生成

核心概念：
  RNN（循环神经网络）与传统 CNN 的区别：
  
    CNN: 一次看一个区域，适合图像（空间数据）          
    RNN: 一次看一个词，适合文本（时序数据）            
    LSTM: RNN 的改进版，解决了"记不住前面内容"的问题  
  

网络结构：
  Embedding  LSTM  Dropout  Linear  Softmax
                                      
  字向量   提取时序   防过拟合   映射到   输出概率
            特征               词表大小
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class PoetryLSTM(nn.Module):
    """
    基于 LSTM 的古诗词生成模型

    模型处理流程：
      输入: "床前明"（三个字的索引 [3, 5, 7]）
      过程: Embedding  LSTM  Linear
      输出: 词表上每个字的概率分布  取概率最高的字

    训练目标：预测下一个字
      "床"  "前"
      "前"  "明"
      "明"  "月"
    """

    def __init__(self, vocab_size, embedding_dim=None,
                 hidden_dim=None, num_layers=None, dropout=None):
        super().__init__()

        if embedding_dim is None:
            embedding_dim = config.EMBEDDING_DIM
        if hidden_dim is None:
            hidden_dim = config.HIDDEN_DIM
        if num_layers is None:
            num_layers = config.NUM_LAYERS
        if dropout is None:
            dropout = config.DROPOUT

        #  词嵌入层 
        # 作用：把离散的"字索引"转为稠密的"字向量"
        # 例：   "床"(索引3)  [0.2, -0.5, 0.8, ...]  (128维向量)
        #       "明"(索引5)  [0.7, 0.1, -0.3, ...]  (128维向量)
        # 这样含义相近的字，向量距离也近
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # <PAD> 的嵌入始终为 0
        )

        #  LSTM 层 
        # LSTM 结构：遗忘门  输入门  输出门
        # 核心思想：用一个"细胞状态"记录长期记忆
        # 对比 RNN：
        #   RNN:   每次看到新词就覆盖旧记忆  记不住前面的
        #   LSTM: 有三个门控制"忘什么"、"记什么"、"输出什么"
        #            能记住几十个字之前的内容
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        #  Dropout 层 
        # 作用：防止过拟合，和 CNN 中一样
        self.dropout = nn.Dropout(dropout)

        #  全连接输出层 
        # 作用：将 LSTM 输出的隐藏状态映射到词表大小
        # 输出 shape: [batch, vocab_size]
        # 每个位置的数值 = 对应"字"的得分
        self.fc = nn.Linear(hidden_dim, vocab_size)

        #  权重初始化 
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, hidden=None):
        """
        前向传播

        参数:
            x: 输入序列，shape = [batch_size, seq_length]
               例: [[3, 5, 7, 9, 11],     "床前明月光"
                    [4, 6, 8, 10, 12]]    "疑是地上霜"
            hidden: LSTM 的初始隐藏状态（训练时=None，生成时传入）

        返回:
            logits: 每个位置每个字的得分
                    shape = [batch_size, seq_length, vocab_size]
            hidden: LSTM 的隐藏状态（用于生成时继续）
        """
        # 1. 词嵌入: 字索引  字向量
        #    [batch, seq_len]  [batch, seq_len, embedding_dim]
        x = self.embedding(x)

        # 2. LSTM 提取时序特征
        #    [batch, seq_len, embedding_dim] 
        #    [batch, seq_len, hidden_dim]
        output, hidden = self.lstm(x, hidden)

        # 3. Dropout 正则化
        output = self.dropout(output)

        # 4. 全连接层映射到词表大小
        #    [batch, seq_len, hidden_dim] 
        #    [batch, seq_len, vocab_size]
        logits = self.fc(output)

        return logits, hidden


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    # 测试模型
    vocab_size = 5000
    model = PoetryLSTM(vocab_size)
    total, trainable = count_parameters(model)
    print(f"参数量: 总计 {total:,}, 可训练 {trainable:,}")

    # 测试前向传播
    x = torch.randint(0, vocab_size, (2, 10))  # 模拟 2 句诗，每句 10 字
    logits, hidden = model(x)
    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {logits.shape}")
    print(f"输出 vocab_size: {logits.shape[-1]} = {vocab_size} ")
