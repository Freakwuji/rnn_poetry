"""
主程序  RNN 古诗词生成：训练 / 测试 / 预测
"""

import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 解决 Windows 终端编码问题
if sys.platform.startswith("win") and hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import config
from model import PoetryLSTM, count_parameters
from process import (load_poetry, build_vocab, save_vocab,
                     load_vocab, prepare_dataset)


#  数据集封装 

class PoetryDataset(Dataset):
    """古诗数据集：为 RNN 语言模型提供成对的（输入，目标）序列"""

    def __init__(self, inputs, targets, seq_length):
        # 将数据切分为等长序列
        # RNN 需要固定长度的输入，所以将诗句分块
        self.seq_length = seq_length
        self.data = []
        for i in range(0, len(inputs) - seq_length, seq_length):
            x = inputs[i:i + seq_length]
            y = targets[i:i + seq_length]
            if len(x) == seq_length:
                self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return (torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long))


#  训练函数 

def train_epoch(model, loader, criterion, optimizer, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    start_time = time.time()

    pbar = loader
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

        optimizer.zero_grad()
        logits, _ = model(inputs)

        # 计算损失（忽略 <PAD> 位置）
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        # 计算准确率
        _, predicted = logits.max(-1)
        mask = targets != 0  # 忽略 <PAD>
        total_correct += (predicted == targets)[mask].sum().item()
        total_tokens += mask.sum().item()

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = total_correct / total_tokens if total_tokens > 0 else 0
    elapsed = time.time() - start_time

    return epoch_loss, epoch_acc, elapsed


def evaluate(model, loader, criterion):
    """验证集评估"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            total_loss += loss.item() * inputs.size(0)
            _, predicted = logits.max(-1)
            mask = targets != 0
            total_correct += (predicted == targets)[mask].sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0
    return avg_loss, avg_acc


def train():
    """完整训练流程"""
    print("=" * 55)
    print("   RNN 古诗词生成  训练")
    print("=" * 55)
    print(f"   设备: {config.DEVICE}")

    # 1. 加载数据
    print("\n[数据] 加载古诗语料...")
    poems = load_poetry()
    if not poems:
        print("[失败] 请先下载古诗数据集放入 data/poetry.txt")
        return

    # 2. 构建词表
    print("\n[词表] 构建词表...")
    try:
        vocab = load_vocab()
    except:
        vocab = build_vocab(poems)
        save_vocab(vocab)

    # 3. 准备数据
    inputs, targets = prepare_dataset(poems, vocab, config.SEQ_LENGTH)
    dataset = PoetryDataset(inputs, targets, config.SEQ_LENGTH)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                               shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=config.NUM_WORKERS)

    print(f"[数据] 训练集: {len(train_dataset)} 样本")
    print(f"[数据] 验证集: {len(val_dataset)} 样本")

    # 4. 创建模型
    print("\n[模型] 创建模型...")
    model = PoetryLSTM(vocab['size']).to(config.DEVICE)
    total_params, trainable_params = count_parameters(model)
    print(f"   参数量: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    print(model)

    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 <PAD>
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 6. 训练循环
    print(f"\n{'='*55}")
    print(f"[开始] 开始训练 ({config.EPOCHS} epochs)")
    print(f"{'='*55}")

    best_loss = float('inf')

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc, elapsed = train_epoch(
            model, train_loader, criterion, optimizer, epoch
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(
            f"Epoch {epoch:2d}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            model_path = os.path.join(config.CKPT_DIR,
                                      f'{config.MODEL_NAME}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab_size': vocab['size'],
                'loss': best_loss,
            }, model_path)

    print(f"\n[OK] 训练完成! 最佳验证损失: {best_loss:.4f}")
    print(f"   模型已保存: {config.CKPT_DIR}/{config.MODEL_NAME}_best.pt")


#  生成函数 

def generate(start_word, model, vocab, max_len=None, temperature=None, top_k=None):
    """
    给定起始字，生成后续诗句

    流程:
      1. 输入起始字  模型预测下一个字
      2. 将预测结果加入输入  继续预测
      3. 重复直到达到最大长度或遇到结束符

    温度参数 temperature:
      低 (0.5)  更保守，总是选概率最高的字  更"工整"
      高 (1.2)  更随机，选概率低的字  更"有创意"
    """
    if max_len is None:
        max_len = config.MAX_GEN_LENGTH
    if temperature is None:
        temperature = config.TEMPERATURE
    if top_k is None:
        top_k = config.TOP_K

    model.eval()
    word2idx = vocab['word2idx']
    idx2word = vocab['idx2word']

    # 将起始字转为索引
    indices = [word2idx.get(w, vocab['unk_idx']) for w in start_word]
    input_tensor = torch.tensor([indices], dtype=torch.long).to(config.DEVICE)

    generated = list(start_word)

    with torch.no_grad():
        # 获取初始隐藏状态
        _, hidden = model(input_tensor)

        # 从最后一个字继续生成
        current = input_tensor[:, -1:]

        for _ in range(max_len):
            logits, hidden = model(current, hidden)

            # 取最后一个位置的输出
            logits = logits[:, -1, :]

            # 温度采样
            logits = logits / temperature

            # Top-k 采样：只从前 k 个候选中选
            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                logits[logits < values[:, -1:]] = float('-inf')

            # 计算概率
            probs = torch.softmax(logits, dim=-1)

            # 从概率分布中采样
            next_idx = torch.multinomial(probs, 1).item()

            # 遇到结束符停止
            if next_idx == vocab['eos_idx']:
                break

            # 将生成的字符转为汉字
            next_char = idx2word.get(next_idx, '？')
            generated.append(next_char)

            # 更新输入
            current = torch.tensor([[next_idx]], dtype=torch.long).to(config.DEVICE)

    return ''.join(generated)


def gen_acrostic(headers, model, vocab):
    """
    生成藏头诗

    例: headers="深度学习"
        "深..." + "度..." + "学..." + "习..."
    """
    model.eval()
    word2idx = vocab['word2idx']
    idx2word = vocab['idx2word']
    unk_idx = vocab['unk_idx']

    result = []
    for header in headers:
        # 生成以该字开头的诗句
        poem = generate(header, model, vocab,
                        max_len=10, temperature=0.8, top_k=8)
        # 取第一句
        if len(poem) > 1:
            result.append(poem[:7])  # 取前 7 字
        else:
            result.append(header + '？')

    return '，'.join(result)


def predict():
    """生成古诗"""
    print("=" * 55)
    print("   古诗词生成")
    print("=" * 55)

    # 加载词表
    try:
        vocab = load_vocab()
    except:
        print("[失败] 词表不存在，请先训练")
        return

    # 加载模型
    model_path = os.path.join(config.CKPT_DIR, f'{config.MODEL_NAME}_best.pt')
    if not os.path.exists(model_path):
        print(f"[失败] 模型不存在: {model_path}")
        print("   请先运行训练: python main.py --train")
        return

    model = PoetryLSTM(vocab['size']).to(config.DEVICE)
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[文件] 模型已加载 (Epoch {checkpoint.get('epoch', '?')})")

    # 生成测试
    print("\n" + "=" * 55)
    print("📝 生成示例")
    print("=" * 55)

    # 示例 1: 给定起始字
    test_starts = ['春江花月夜', '床前明月光', '白日依山尽',
                   '深度学习', '人工智能']
    for start in test_starts:
        result = generate(start, model, vocab)
        print(f"\n   '{start}'  {result}")

    # 示例 2: 藏头诗
    print("\n\n📝 藏头诗示例")
    headers = '黎坤朗'
    result = gen_acrostic(headers, model, vocab)
    print(f"   '{headers}'  {result}\n")


#  主入口 

def main():
    import argparse

    parser = argparse.ArgumentParser(description="RNN 古诗词生成")
    parser.add_argument('--train', action='store_true', help='训练模式')
    parser.add_argument('--predict', action='store_true', help='生成模式')
    parser.add_argument('--start', type=str, default='春江',
                        help='起始字（生成模式）')
    parser.add_argument('--acrostic', type=str, default='',
                        help='藏头诗（生成模式）')

    args = parser.parse_args()

    if args.train:
        train()
    elif args.predict:
        predict()
    elif args.acrostic:
        try:
            vocab = load_vocab()
            model_path = os.path.join(config.CKPT_DIR,
                                      f'{config.MODEL_NAME}_best.pt')
            model = PoetryLSTM(vocab['size']).to(config.DEVICE)
            checkpoint = torch.load(model_path, map_location=config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            result = gen_acrostic(args.acrostic, model, vocab)
            print(f"藏头诗 '{args.acrostic}': {result}")
        except Exception as e:
            print(f"[失败] 生成失败: {e}")
    else:
        # 默认：先预测，给出提示
        try:
            predict()
        except:
            print("请先运行: python main.py --train")
            print("然后运行: python main.py --predict")


if __name__ == '__main__':
    main()
