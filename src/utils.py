"""
工具函数
"""

import json
import torch
import config


def save_checkpoint(model, optimizer, epoch, loss, filename):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)
    print(f"💾 检查点已保存: {filename}")


def load_checkpoint(model, optimizer, filename):
    """加载检查点"""
    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    print(f"📂 已加载检查点: {filename} (Epoch {epoch})")
    return epoch, loss


def set_seed(seed=42):
    """设置随机种子，保证结果可复现"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
