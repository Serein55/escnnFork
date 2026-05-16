#!/usr/bin/env python3
"""
一键运行 P4CNN Rotated MNIST 实验

功能:
1. 检查依赖
2. 测试数据加载
3. 测试模型定义
4. 训练模型
5. 评估结果
"""

import os
import sys
import subprocess
import time
import argparse


def check_dependencies():
    """检查必要的依赖"""
    print("=" * 60)
    print("检查依赖...")
    print("=" * 60)
    
    required_packages = ['torch', 'torchvision', 'escnn', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下包: {', '.join(missing_packages)}")
        print("请使用以下命令安装:")
        print("pip install torch torchvision numpy")
        print("cd /path/to/escnn && pip install -e .")
        return False
    
    print("\n所有依赖已安装!\n")
    return True


def test_dataset():
    """测试数据集加载"""
    print("=" * 60)
    print("测试数据集加载...")
    print("=" * 60)
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dataset import get_dataloaders
        
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=100,
            num_workers=2
        )
        
        # 测试一个批次
        images, labels = next(iter(train_loader))
        print(f"训练集批次形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print("数据集加载测试通过!\n")
        return True
        
    except Exception as e:
        print(f"数据集加载测试失败: {e}\n")
        return False


def test_model():
    """测试模型定义"""
    print("=" * 60)
    print("测试 P4CNN 模型...")
    print("=" * 60)
    
    try:
        import torch
        from p4cnn import P4CNN, count_parameters
        
        # 创建模型
        model = P4CNN(num_classes=10, n_channels=10)
        
        # 统计参数
        total_params, trainable_params = count_parameters(model)
        
        # 测试前向传播
        x = torch.randn(4, 1, 28, 28)
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print("模型测试通过!\n")
        return True
        
    except Exception as e:
        print(f"模型测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def train_model(args):
    """训练模型"""
    print("=" * 60)
    print("开始训练 P4CNN...")
    print("=" * 60)
    
    # 构建训练命令
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py'),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--n_channels', str(args.n_channels),
        '--save_dir', args.save_dir,
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    # 运行训练
    start_time = time.time()
    result = subprocess.run(cmd)
    total_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n训练完成! 总耗时: {total_time:.2f} 秒")
        return True
    else:
        print("\n训练失败!")
        return False


def run_full_experiment(args):
    """运行完整实验"""
    print("\n" + "#" * 60)
    print("#  P4CNN Rotated MNIST 实验")
    print("#" * 60 + "\n")
    
    total_start = time.time()
    
    # 1. 检查依赖
    if not check_dependencies():
        return
    
    # 2. 测试数据集
    if not args.skip_tests:
        if not test_dataset():
            return
        
        # 3. 测试模型
        if not test_model():
            return
    
    # 4. 训练模型
    if not train_model(args):
        return
    
    total_time = time.time() - total_start
    
    print("\n" + "#" * 60)
    print(f"#  实验完成!")
    print(f"#  总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print("#" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='P4CNN Rotated MNIST 一键实验')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数 (default: 200)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小 (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率 (default: 0.001)')
    parser.add_argument('--n_channels', type=int, default=10,
                        help='P4 通道数 (default: 10)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='保存目录 (default: ./checkpoints)')
    
    # 测试参数
    parser.add_argument('--skip_tests', action='store_true',
                        help='跳过数据集和模型测试')
    
    args = parser.parse_args()
    
    # 运行实验
    run_full_experiment(args)


if __name__ == '__main__':
    main()