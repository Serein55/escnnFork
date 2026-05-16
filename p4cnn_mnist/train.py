"""
Rotated MNIST 训练脚本
支持 P4CNN 与参数量匹配的传统 CNN 对照（`--model baseline`）。

超参数设置：
- 优化器：Adam
- 损失函数：Cross-Entropy Loss
- 训练集：10,000 样本
- 验证集：2,000 样本 (用于模型选择)
- 测试集：10,000 样本 (论文原始设置)
- 目标错误率：约 2.28%
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloaders
from p4cnn import P4CNN, count_parameters
from baseline_cnn import make_baseline_matched_to_p4cnn


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        print(f'[{batch_idx}/{len(train_loader)}] '
                f'Loss: {loss.item():.4f} | '
                f'Acc: {100. * correct / total:.2f}%')
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc


def evaluate(model, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    loss = running_loss / len(data_loader)
    accuracy = 100. * correct / total
    error_rate = 100. - accuracy
    
    return loss, accuracy, error_rate


def train(args):
    """主训练函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据
    print("\n" + "="*60)
    print("加载 Rotated MNIST 数据集...")
    print("="*60)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers
    )
    
    # 创建模型
    print("\n" + "="*60)
    if args.model == "p4cnn":
        print("创建 P4CNN 模型...")
        print("="*60)
        model = P4CNN(
            num_classes=10,
            n_channels=args.n_channels
        ).to(device)
        baseline_width = None
        baseline_width_last = None
        ref_p4_param_count = None
    else:
        print("创建对照 BaselineCNN（参数量对齐 P4CNN）...")
        print("="*60)
        model, baseline_width, baseline_width_last, ref_p4_param_count, baseline_param_count = (
            make_baseline_matched_to_p4cnn(
                num_classes=10,
                n_channels=args.n_channels,
            )
        )
        model = model.to(device)
        wl_note = (
            f", 第7层输出通道 width_last={baseline_width_last}"
            if baseline_width_last != baseline_width
            else ""
        )
        print(
            f"参照 P4CNN(n_channels={args.n_channels}) 参数量: {ref_p4_param_count:,}\n"
            f"BaselineCNN width={baseline_width}{wl_note} 参数量: {baseline_param_count:,} "
            f"(差值 {baseline_param_count - ref_p4_param_count:+d})"
        )

    total_params, trainable_params = count_parameters(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # 训练记录
    best_val_acc = 0.0
    best_val_error = 100.0
    best_epoch = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60)
    print(f"训练配置:")
    print(f"  模型: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"  P4 参照通道数 n_channels: {args.n_channels}")
    if args.model == "baseline" and baseline_width is not None:
        print(f"  Baseline 宽度 width / width_last: {baseline_width} / {baseline_width_last}")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print("-" * 40)
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        val_loss, val_acc, val_error = evaluate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\n训练 - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | Error: {val_error:.2f}%")
        print(f"当前学习率: {current_lr:.6f}")
        
        # 保存最佳模型 (基于验证集性能)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_error = val_error
            best_epoch = epoch
            
            # 保存模型
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model': args.model,
                'n_channels': args.n_channels,
                'baseline_width': baseline_width,
                'baseline_width_last': baseline_width_last,
                'ref_p4_param_count': ref_p4_param_count,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_error': val_error,
                'train_loss': train_loss,
            }, save_path)
            print(f"✓ 保存最佳模型 (验证错误率: {val_error:.2f}%)")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"总训练时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"最佳验证性能 - Epoch {best_epoch}:")
    print(f"  验证准确率: {best_val_acc:.2f}%")
    print(f"  验证错误率: {best_val_error:.2f}%")
    
    # 加载最佳模型进行测试
    print("\n" + "="*60)
    print("在测试集上评估最佳模型...")
    print("="*60)
    
    checkpoint = torch.load(
        os.path.join(args.save_dir, 'best_model.pth'),
        map_location=device,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_error = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\n测试集结果:")
    print(f"  测试准确率: {test_acc:.2f}%")
    print(f"  测试错误率: {test_error:.2f}%")
    print(f"  测试损失: {test_loss:.4f}")
    
    print("\n" + "="*60)
    if args.model == "p4cnn":
        print("与论文对比:")
        print("="*60)
        print(f"  论文 P4CNN 目标错误率: 2.28%")
        print(f"  实际测试错误率: {test_error:.2f}%")
        if test_error <= 2.28:
            print("  ✓ 达到论文目标!")
        else:
            print(f"  ✗ 与目标差距: {test_error - 2.28:.2f}%")
    else:
        print("对照实验说明:")
        print("="*60)

        print(f"  本 Baseline 测试错误率: {test_error:.2f}%（可与同设置 P4CNN 直接对比）。")
    
    # 保存训练历史
    history = {
        'model': args.model,
        'n_channels': args.n_channels,
        'baseline_width': baseline_width,
        'baseline_width_last': baseline_width_last,
        'ref_p4_param_count': ref_p4_param_count,
        'trainable_param_count': int(trainable_params),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_error': test_error,
    }
    
    import json
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\n训练历史已保存到: {history_path}")
    
    return test_error


def main():
    parser = argparse.ArgumentParser(description='P4CNN 训练 Rotated MNIST')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据集存储目录')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作线程数')
    
    # 模型参数
    parser.add_argument(
        '--model', type=str, default='p4cnn', choices=('p4cnn', 'baseline'),
        help='p4cnn: 等变网络; baseline: 传统 CNN，总参数量对齐当前 n_channels 下的 P4CNN',
    )
    parser.add_argument('--n_channels', type=int, default=10,
                        help='P4CNN 通道数；baseline 模式下用于选定与之对齐的参照 P4CNN 参数量')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减 (L2 正则化)')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    
    args = parser.parse_args()
    
    # 开始训练
    test_error = train(args)
    
    return test_error


if __name__ == '__main__':
    main()