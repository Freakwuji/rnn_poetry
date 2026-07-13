"""
改进的训练器

包含：
1. Warmup + CosineAnnealing 学习率调度
2. 早停机制
3. 混合精度训练（可选）
4. 训练日志记录
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import config


class EarlyStopping:
    """
    早停机制

    当验证集 loss 不再下降时，提前停止训练
    避免过拟合，节省训练时间
    """

    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        """
        参数:
            patience: 容忍多少个 epoch 没有改善
            min_delta: 最小改善幅度
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch, model, save_path):
        """
        检查是否应该早停

        参数:
            val_loss: 当前验证集 loss
            epoch: 当前 epoch
            model: 模型
            save_path: 保存路径

        返回:
            early_stop: 是否应该停止
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, epoch, model, save_path)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience} (best: {self.best_loss:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  EarlyStopping: 在 epoch {epoch} 停止训练")
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, epoch, model, save_path)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, epoch, model, save_path):
        """保存最佳模型"""
        if self.verbose:
            print(f"  保存最佳模型 (loss: {val_loss:.4f}, epoch: {epoch})")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }, save_path)


class WarmupCosineScheduler:
    """
    Warmup + CosineAnnealing 学习率调度

    阶段：
    1. Warmup 阶段：学习率从 0 线性增加到 max_lr
    2. Cosine 阶段：学习率按余弦函数衰减到 min_lr

    为什么这样设计：
    - Warmup：避免初期梯度不稳定
    - Cosine：比 StepLR 更平滑，训练更稳定
    """

    def __init__(self, optimizer, max_lr, warmup_epochs, total_epochs, min_lr=None):
        """
        参数:
            optimizer: 优化器
            max_lr: 最大学习率
            warmup_epochs: warmup 持续的 epoch 数
            total_epochs: 总 epoch 数
            min_lr: 最小学习率（默认为 max_lr 的 1/100）
        """
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr or max_lr / 100
        self.current_epoch = 0

    def step(self):
        """更新学习率"""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Warmup 阶段：线性增加
            lr = self.max_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine 阶段：余弦衰减
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


class Trainer:
    """
    改进的训练器

    功能：
    1. 支持多种模型（LSTM, LSTM+Attention, GRU+Attention 等）
    2. Warmup + CosineAnnealing 学习率调度
    3. 早停机制
    4. 混合精度训练（可选）
    5. 训练日志记录
    """

    def __init__(self, model, vocab, model_name='model', use_amp=False):
        """
        参数:
            model: 模型
            vocab: 词表
            model_name: 模型名称（用于保存）
            use_amp: 是否使用混合精度训练
        """
        self.model = model
        self.vocab = vocab
        self.model_name = model_name
        self.use_amp = use_amp and torch.cuda.is_available()

        # 损失函数（忽略 padding）
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # 混合精度训练
        self.scaler = GradScaler() if self.use_amp else None

        # 训练日志
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': [],
        }

    def train_epoch(self, train_loader, optimizer, scheduler=None):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            optimizer.zero_grad()

            # 混合精度训练
            if self.use_amp:
                with autocast():
                    logits, _ = self.model(inputs)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                logits, _ = self.model(inputs)
                loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)

            # 计算准确率
            _, predicted = logits.max(-1)
            mask = targets != 0
            total_correct += (predicted == targets)[mask].sum().item()
            total_tokens += mask.sum().item()

            # 更新学习率
            if scheduler is not None:
                scheduler.step()

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct / total_tokens if total_tokens > 0 else 0
        elapsed = time.time() - start_time

        return epoch_loss, epoch_acc, elapsed

    def evaluate(self, val_loader):
        """验证集评估"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(config.DEVICE)
                targets = targets.to(config.DEVICE)

                logits, _ = self.model(inputs)
                loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                total_loss += loss.item() * inputs.size(0)
                _, predicted = logits.max(-1)
                mask = targets != 0
                total_correct += (predicted == targets)[mask].sum().item()
                total_tokens += mask.sum().item()

        avg_loss = total_loss / len(val_loader.dataset)
        avg_acc = total_correct / total_tokens if total_tokens > 0 else 0

        return avg_loss, avg_acc

    def train(self, train_loader, val_loader, epochs=200, lr=0.001,
              warmup_epochs=5, patience=15, save_dir='checkpoints'):
        """
        完整训练流程

        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 总训练轮数
            lr: 最大学习率
            warmup_epochs: warmup 轮数
            patience: 早停容忍度
            save_dir: 模型保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{self.model_name}_best.pt')

        print(f"\n{'='*60}")
        print(f"  开始训练: {self.model_name}")
        print(f"{'='*60}")
        print(f"  设备: {config.DEVICE}")
        print(f"  混合精度: {'是' if self.use_amp else '否'}")
        print(f"  总轮数: {epochs}")
        print(f"  学习率: {lr}")
        print(f"  Warmup: {warmup_epochs} 轮")
        print(f"  早停: {patience} 轮")
        print(f"{'='*60}\n")

        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 学习率调度器
        scheduler = WarmupCosineScheduler(
            optimizer,
            max_lr=lr,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs
        )

        # 早停
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        # 训练循环
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_acc, elapsed = self.train_epoch(
                train_loader, optimizer, scheduler
            )

            # 验证
            val_loss, val_acc = self.evaluate(val_loader)

            # 获取当前学习率
            current_lr = scheduler.get_lr()

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # 打印信息
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # 早停检查
            if early_stopping(val_loss, epoch, self.model, save_path):
                print(f"\n早停触发！最佳 epoch: {early_stopping.best_epoch}")
                break

        total_time = time.time() - start_time
        print(f"\n训练完成! 总时间: {total_time:.1f}s")
        print(f"最佳模型已保存: {save_path}")

        # 保存训练历史
        history_path = os.path.join(save_dir, f'{self.model_name}_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"训练历史已保存: {history_path}")

        return self.history

    def load_best_model(self, save_dir='checkpoints'):
        """加载最佳模型"""
        save_path = os.path.join(save_dir, f'{self.model_name}_best.pt')
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location=config.DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载最佳模型 (epoch {checkpoint['epoch']})")
            return checkpoint
        else:
            print(f"未找到模型: {save_path}")
            return None
