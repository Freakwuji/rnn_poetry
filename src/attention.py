"""
Attention 机制实现

为什么需要 Attention：
- 基础 LSTM 只用最后一个时间步的输出，丢失了前面的信息
- Attention 让模型在生成每个字时，能"回头看"前面所有位置
- 对古诗生成特别重要：需要关注押韵、对仗等长距离依赖

为什么选 Bahdanau Attention：
- Dot-Product Attention 要求 query 和 key 维度相同
- Bahdanau Attention 更灵活，可以学习不同维度之间的对齐
- 适合古诗生成这种需要精细控制的任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention（加性注意力）

    核心思想：
      query（当前状态）和 keys（所有历史状态）做对齐
      得到 attention weights
      用 weights 对 keys 加权求和得到 context

    公式：
      score(query, key) = v^T * tanh(W_a * query + U_a * key)
      attention_weights = softmax(score)
      context = sum(attention_weights * keys)
    """

    def __init__(self, hidden_dim):
        """
        参数:
            hidden_dim: LSTM 隐藏层维度
        """
        super().__init__()

        # 注意力参数
        # W_a: 作用在 query 上
        # U_a: 作用在 keys 上
        # v_a: 最后映射到标量
        self.Wa = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Ua = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.va = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, keys, mask=None):
        """
        前向传播

        参数:
            query: 当前时间步的隐藏状态
                   shape = [batch_size, 1, hidden_dim]
            keys: 所有时间步的输出
                   shape = [batch_size, seq_len, hidden_dim]
            mask: 可选的 mask（忽略 padding 位置）

        返回:
            context: 加权求和后的上下文向量
                     shape = [batch_size, 1, hidden_dim]
            weights: 注意力权重
                     shape = [batch_size, seq_len, 1]
        """
        # 计算注意力分数
        # query: [batch, 1, hidden] -> [batch, 1, hidden]
        # keys: [batch, seq_len, hidden] -> [batch, seq_len, hidden]
        query_transformed = self.Wa(query)  # [batch, 1, hidden]
        keys_transformed = self.Ua(keys)    # [batch, seq_len, hidden]

        # 广播相加
        # [batch, 1, hidden] + [batch, seq_len, hidden] -> [batch, seq_len, hidden]
        combined = query_transformed + keys_transformed

        # 计算分数
        # [batch, seq_len, hidden] -> [batch, seq_len, 1]
        scores = self.va(torch.tanh(combined))

        # 应用 mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 归一化得到权重
        # [batch, seq_len, 1] -> [batch, seq_len, 1]
        weights = F.softmax(scores, dim=1)

        # 加权求和
        # weights: [batch, seq_len, 1]
        # keys: [batch, seq_len, hidden]
        # context: [batch, 1, hidden]
        context = torch.bmm(weights.transpose(1, 2), keys)

        return context, weights


class LuongAttention(nn.Module):
    """
    Luong Attention（点积注意力）

    比 Bahdanau 更简单，计算更快
    但要求 query 和 key 维度相同

    公式：
      score(query, key) = query^T * key
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.scale = hidden_dim ** 0.5

    def forward(self, query, keys, mask=None):
        """
        参数:
            query: [batch_size, 1, hidden_dim]
            keys: [batch_size, seq_len, hidden_dim]
            mask: 可选

        返回:
            context: [batch_size, 1, hidden_dim]
            weights: [batch_size, seq_len, 1]
        """
        # 计算点积分数
        # [batch, 1, hidden] x [batch, hidden, seq_len] -> [batch, 1, seq_len]
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)  # [batch, 1, seq_len]

        context = torch.bmm(weights, keys)  # [batch, 1, hidden]

        return context, weights.transpose(1, 2)


class MultiHeadAttention(nn.Module):
    """
    多头注意力（Transformer 中使用）

    优势：
    - 可以同时关注不同位置的信息
    - 每个头学习不同的注意力模式
    - 更强大，但参数更多

    用于对比实验
    """

    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)
        self.Wo = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, keys, mask=None):
        batch_size = query.size(0)

        # 线性变换
        Q = self.Wq(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(keys).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(keys).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.Wo(context)

        return output, weights.mean(dim=1)  # 平均所有头的权重
