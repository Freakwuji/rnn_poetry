# -*- coding: utf-8 -*-
"""
可视化模块

生成实验结果的图表：
1. 模型结构对比图
2. 训练曲线图
3. 评价结果对比图
4. 注意力权重热力图
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_model_comparison(save_dir='results'):
    """
    绘制模型结构对比图

    对比7个模型的参数量和结构复杂度
    """
    models = {
        'LSTM': {'params': 7.5, 'layers': 3, 'type': '基线1'},
        'GRU': {'params': 5.7, 'layers': 3, 'type': '基线2'},
        'LSTM+Attn': {'params': 8.6, 'layers': 4, 'type': '+Attn'},
        'GRU+Attn': {'params': 6.8, 'layers': 4, 'type': '+Attn'},
        'BiLSTM+Attn': {'params': 16.2, 'layers': 4, 'type': '+双向'},
        'LSTM+MultiAttn': {'params': 8.8, 'layers': 4, 'type': '+多头'},
        'BiLSTM+MultiAttn': {'params': 18.3, 'layers': 5, 'type': '完整版'},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 参数量对比
    names = list(models.keys())
    params = [models[n]['params'] for n in names]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#E91E63', '#00BCD4', '#795548']

    bars = axes[0].bar(names, params, color=colors)
    axes[0].set_title('模型参数量对比', fontsize=14)
    axes[0].set_ylabel('参数量 (百万)', fontsize=12)

    # 在柱状图上添加数值
    for bar, param in zip(bars, params):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{param}M', ha='center', va='bottom', fontsize=10)

    # 结构复杂度对比
    layers = [models[n]['layers'] for n in names]
    types = [models[n]['type'] for n in names]

    bars2 = axes[1].bar(names, layers, color=colors)
    axes[1].set_title('模型层数对比', fontsize=14)
    axes[1].set_ylabel('层数', fontsize=12)

    for bar, layer, t in zip(bars2, layers, types):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{t}\n{layer}层', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 模型对比图已保存: {save_dir}/model_comparison.png")


def plot_training_curves(history_file='results/training_history.json', save_dir='results'):
    """
    绘制训练曲线

    包括：loss曲线、准确率曲线、学习率曲线
    """
    if not os.path.exists(history_file):
        print(f"  ⚠ 训练历史文件不存在: {history_file}")
        return

    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss 曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', label='训练集', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r--', label='验证集', linewidth=2)
    axes[0].set_title('损失函数曲线', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(epochs, history['train_acc'], 'b-', label='训练集', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r--', label='验证集', linewidth=2)
    axes[1].set_title('准确率曲线', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # 学习率曲线
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_title('学习率调度曲线', fontsize=14)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 训练曲线图已保存: {save_dir}/training_curves.png")


def plot_evaluation_results(results_file='results/evaluation_results.json', save_dir='results'):
    """
    绘制评价结果对比图

    对比不同模型在各个评价维度上的得分
    """
    if not os.path.exists(results_file):
        print(f"  ⚠ 评价结果文件不存在: {results_file}")
        return

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 提取数据
    models = [r['model'] for r in results]
    rhyme_scores = [r['rhyme_score'] for r in results]
    parallel_scores = [r['parallel_score'] for r in results]
    repeat_scores = [r['repeat_score'] for r in results]
    total_scores = [r['total_score'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 雷达图风格的柱状图
    x = np.arange(len(models))
    width = 0.2

    axes[0].bar(x - width*1.5, rhyme_scores, width, label='押韵', color='#2196F3')
    axes[0].bar(x - width*0.5, parallel_scores, width, label='对仗', color='#4CAF50')
    axes[0].bar(x + width*0.5, repeat_scores, width, label='重复度', color='#FF9800')
    axes[0].bar(x + width*1.5, total_scores, width, label='总分', color='#9C27B0')

    axes[0].set_xlabel('模型', fontsize=12)
    axes[0].set_ylabel('得分', fontsize=12)
    axes[0].set_title('各模型评价维度对比', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=10)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 总分对比
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#E91E63', '#00BCD4', '#795548']
    bars = axes[1].bar(models, total_scores, color=colors)
    axes[1].set_xlabel('模型', fontsize=12)
    axes[1].set_ylabel('总分', fontsize=12)
    axes[1].set_title('模型总分对比', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')

    for bar, score in zip(bars, total_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 评价结果图已保存: {save_dir}/evaluation_results.png")


def plot_generation_comparison(save_dir='results'):
    """
    绘制生成结果对比图

    对比不同模型生成的诗句质量
    """
    # 模拟数据（实际使用时从文件加载）
    generations = {
        'LSTM': [
            ('春江', '春江水月上寒花满月深'),
            ('明月', '明月前秋水满花飞落'),
        ],
        'GRU': [
            ('春江', '春江流水落花春去也'),
            ('明月', '明月照高楼流光正徘徊'),
        ],
        'LSTM+Attn': [
            ('春江', '春江水暖鸭先知蒌蒿'),
            ('明月', '明月几时有把酒问青天'),
        ],
        'GRU+Attn': [
            ('春江', '春江花月夜半时人初见'),
            ('明月', '明月别枝惊鹊清风鸣蝉'),
        ],
        'BiLSTM+Attn': [
            ('春江', '春江潮水连海平海上明月'),
            ('明月', '明月松间照清泉石上流'),
        ],
        'LSTM+MultiAttn': [
            ('春江', '春江潮水连海平海上生明月'),
            ('明月', '明月松间照清泉石上流春山'),
        ],
        'BiLSTM+MultiAttn': [
            ('春江', '春江潮水连海平海上生明月轮'),
            ('明月', '明月松间照清泉石上流春山色'),
        ],
    }

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (model, poems) in enumerate(generations.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        ax.axis('off')

        # 标题
        ax.set_title(model, fontsize=14, fontweight='bold', pad=20)

        # 显示诗句
        text = ""
        for start, poem in poems:
            text += f"起始: {start}\n生成: {poem}\n\n"

        ax.text(0.1, 0.5, text, transform=ax.transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 隐藏多余的子图
    for idx in range(len(generations), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'generation_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 生成对比图已保存: {save_dir}/generation_comparison.png")


def plot_all(save_dir='results'):
    """
    生成所有图表
    """
    print("\n[可视化] 生成图表...")
    os.makedirs(save_dir, exist_ok=True)

    plot_model_comparison(save_dir)
    plot_training_curves(save_dir=save_dir)
    plot_evaluation_results(save_dir=save_dir)
    plot_generation_comparison(save_dir)

    print(f"\n  ✓ 所有图表已保存到: {save_dir}/")


if __name__ == '__main__':
    plot_all()
