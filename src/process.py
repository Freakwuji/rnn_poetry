"""
数据预处理  古诗词语料加载与词表构建
"""

import json
import re
import os
from collections import Counter
import config


def load_poetry(file_path=None):
    """
    从文件加载古诗词语料

    每行一首诗，格式："题目标题:诗句内容"
    过滤：只保留 5 言或 7 言的唐诗
    返回: 诗句列表，每句以空格分词
    """
    if file_path is None:
        file_path = config.DATA_PATH

    if not os.path.exists(file_path):
        print(f"️ 语料文件不存在: {file_path}")
        print("请将古诗词数据集放入 data/poetry.txt")
        print("格式要求：每行一首诗，\"标题:诗句内容\"")
        return []

    poems = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 提取诗句内容（去掉标题）
        if ':' in line:
            content = line.split(':', 1)[1].strip()
        else:
            content = line

        # 过滤非诗句（含特殊字符）
        if not content or not re.match(r'^[一-鿿，。！？、；：\s]+$', content):
            continue

        # 提取每句（按标点分割）
        sentences = re.split(r'[，。！？、；：]', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 4]

        for s in sentences:
            # 只保留 5 言或 7 言
            if len(s) in (5, 7):
                poems.append(s)

    print(f"[数据] 共加载 {len(poems)} 句古诗")
    print(f"   五言: {sum(1 for p in poems if len(p)==5)} 句")
    print(f"   七言: {sum(1 for p in poems if len(p)==7)} 句")
    return poems


def build_vocab(poems, min_freq=2):
    """
    从诗句列表构建词表

    参数:
        poems: 诗句列表
        min_freq: 最小出现频率，低于此频率的字会被忽略

    返回:
        vocab: 词表对象
    """
    # 统计字频
    counter = Counter()
    for poem in poems:
        counter.update(poem)

    # 构建词表
    # 特殊标记：
    #   <PAD>    填充（使批次内序列等长）
    #   <UNK>    未知字
    #   <SOS>    句子起始
    #   <EOS>    句子结束
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']

    # 过滤低频字
    vocab_words = [w for w, c in counter.items() if c >= min_freq]
    vocab_words = sorted(vocab_words)

    # 构建映射
    word2idx = {w: i for i, w in enumerate(special_tokens + vocab_words)}
    idx2word = {i: w for w, i in word2idx.items()}

    vocab = {
        'size': len(word2idx),
        'word2idx': word2idx,
        'idx2word': idx2word,
        'pad_idx': word2idx['<PAD>'],
        'unk_idx': word2idx['<UNK>'],
        'sos_idx': word2idx['<SOS>'],
        'eos_idx': word2idx['<EOS>'],
    }

    print(f"[词表] 词表大小: {vocab['size']} 字")
    print(f"   其中 {len(vocab_words)} 个有效汉字")

    return vocab


def save_vocab(vocab, path=None):
    """保存词表到 JSON"""
    if path is None:
        path = config.VOCAB_PATH

    # 转换 idx2word 的键为字符串（JSON 要求）
    save_dict = {
        'size': vocab['size'],
        'word2idx': vocab['word2idx'],
        'idx2word': {str(k): v for k, v in vocab['idx2word'].items()},
        'pad_idx': vocab['pad_idx'],
        'unk_idx': vocab['unk_idx'],
        'sos_idx': vocab['sos_idx'],
        'eos_idx': vocab['eos_idx'],
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, ensure_ascii=False, indent=2)
    print(f"[保存] 词表已保存: {path}")


def load_vocab(path=None):
    """从 JSON 加载词表"""
    if path is None:
        path = config.VOCAB_PATH

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vocab = {
        'size': data['size'],
        'word2idx': data['word2idx'],
        'idx2word': {int(k): v for k, v in data['idx2word'].items()},
        'pad_idx': data['pad_idx'],
        'unk_idx': data['unk_idx'],
        'sos_idx': data['sos_idx'],
        'eos_idx': data['eos_idx'],
    }
    print(f"[文件] 已加载词表 ({vocab['size']} 字)")
    return vocab


def prepare_dataset(poems, vocab, seq_length=None):
    """
    将诗句转为训练用的序列数据

    RNN 语言模型的任务：给定前 n-1 个字，预测第 n 个字

    例如诗句 "床前明月光"：
       输入: 床前明月  预测: 前明月光
                       
       训练样本: (input="床前明", target="前明月")
                (input="前明月", target="明月光")
    """
    if seq_length is None:
        seq_length = config.SEQ_LENGTH

    inputs = []
    targets = []

    word2idx = vocab['word2idx']
    unk_idx = vocab['unk_idx']
    pad_idx = vocab['pad_idx']

    for poem in poems:
        # 将字转为索引
        indices = [word2idx.get(w, unk_idx) for w in poem]

        # 滑动窗口生成训练样本
        for i in range(len(indices) - 1):
            # 输入: indices[i]
            # 目标: indices[i+1]
            inputs.append(indices[i])
            targets.append(indices[i+1])

    print(f"[目标] 生成 {len(inputs)} 个训练样本")
    return inputs, targets


if __name__ == '__main__':
    # 测试
    poems = load_poetry()
    if poems:
        vocab = build_vocab(poems)
        save_vocab(vocab)
        prepare_dataset(poems, vocab)
