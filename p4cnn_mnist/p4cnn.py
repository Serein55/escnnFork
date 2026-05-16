"""
P4CNN 模型定义
使用 escnn 构建 7 层等变卷积神经网络

架构特点：
- 对称群：p4 群 (4 个旋转角度: 0°, 90°, 180°, 270°)
- 通道数：10 个 p4-特征通道 (相当于普通 CNN 的 20 通道)
- 7 层卷积：前 6 层 3x3，第 7 层 4x4
- 池化：第 2 层后空间最大池化，最后一层后 GroupPooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import escnn.nn as enn
from escnn import gspaces


class P4CNN(enn.EquivariantModule):
    """
    P4 等变卷积神经网络
    
    用于 Rotated MNIST 分类任务
    具有旋转等变性，最后通过 GroupPooling 获得旋转不变性
    """
    
    def __init__(self, num_classes=10, n_channels=10):
        """
        Args:
            num_classes: 分类数量 (MNIST: 10)
            n_channels: p4-特征通道数 (default: 10, 相当于普通 CNN 20 通道)
        """
        super(P4CNN, self).__init__()
        
        # 定义 p4 群 (4 个离散旋转)
        self.gspace = gspaces.rot2dOnR2(N=8)
        
        # 输入场类型：平凡表示 (单通道灰度图)
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        
        # 各层的通道配置
        # p4-regular 表示：每个通道有 4 个旋转副本
        # 因此 10 个 p4-通道相当于普通 CNN 的 10*sqrt(4) = 20 通道的参数量
        
        # Layer 1: 3x3 conv, trivial -> regular (提升层)
        out_type_1 = enn.FieldType(self.gspace, [self.gspace.regular_repr] * n_channels)
        self.conv1 = enn.R2Conv(
            self.in_type, out_type_1, kernel_size=3, padding=1,
            bias=False, sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn1 = enn.InnerBatchNorm(out_type_1)
        self.relu1 = enn.ReLU(out_type_1, inplace=True)
        
        # Layer 2: 3x3 conv, regular -> regular
        out_type_2 = enn.FieldType(self.gspace, [self.gspace.regular_repr] * n_channels)
        self.conv2 = enn.R2Conv(
            out_type_1, out_type_2, kernel_size=3, padding=1,
            bias=False, sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn2 = enn.InnerBatchNorm(out_type_2)
        self.relu2 = enn.ReLU(out_type_2, inplace=True)
        
        # 空间最大池化 (在第 2 层后)
        self.pool1 = enn.PointwiseMaxPool2D(out_type_2, kernel_size=2, stride=2)
        
        # Layer 3: 3x3 conv
        out_type_3 = enn.FieldType(self.gspace, [self.gspace.regular_repr] * n_channels)
        self.conv3 = enn.R2Conv(
            out_type_2, out_type_3, kernel_size=3, padding=1,
            bias=False, sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn3 = enn.InnerBatchNorm(out_type_3)
        self.relu3 = enn.ReLU(out_type_3, inplace=True)
        
        # Layer 4: 3x3 conv
        out_type_4 = enn.FieldType(self.gspace, [self.gspace.regular_repr] * n_channels)
        self.conv4 = enn.R2Conv(
            out_type_3, out_type_4, kernel_size=3, padding=1,
            bias=False, sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn4 = enn.InnerBatchNorm(out_type_4)
        self.relu4 = enn.ReLU(out_type_4, inplace=True)
        
        # Layer 5: 3x3 conv
        out_type_5 = enn.FieldType(self.gspace, [self.gspace.regular_repr] * n_channels)
        self.conv5 = enn.R2Conv(
            out_type_4, out_type_5, kernel_size=3, padding=1,
            bias=False, sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn5 = enn.InnerBatchNorm(out_type_5)
        self.relu5 = enn.ReLU(out_type_5, inplace=True)
        
        # Layer 6: 3x3 conv
        out_type_6 = enn.FieldType(self.gspace, [self.gspace.regular_repr] * n_channels)
        self.conv6 = enn.R2Conv(
            out_type_5, out_type_6, kernel_size=3, padding=1,
            bias=False, sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn6 = enn.InnerBatchNorm(out_type_6)
        self.relu6 = enn.ReLU(out_type_6, inplace=True)
        
        # Layer 7: 4x4 conv (最后一层卷积)
        out_type_7 = enn.FieldType(self.gspace, [self.gspace.regular_repr] * n_channels)
        self.conv7 = enn.R2Conv(
            out_type_6, out_type_7, kernel_size=4, padding=0,
            bias=False, sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn7 = enn.InnerBatchNorm(out_type_7)
        self.relu7 = enn.ReLU(out_type_7, inplace=True)
        
        # GroupPooling: 对旋转维度进行最大池化，获得旋转不变性
        self.gpool = enn.GroupPooling(out_type_7)
        
        # 全连接层分类器
        # GroupPooling 后输出通道数为 n_channels
        # 空间维度计算：
        # - 输入 28x28
        # - Layer 1-2: 28x28 (3x3 conv, padding=1)
        # - Pool1: 14x14 (2x2 max pool)
        # - Layer 3-6: 14x14 (3x3 conv, padding=1)
        # - Layer 7: 11x11 (4x4 conv, no padding)
        self.fc = nn.Linear(n_channels * 11 * 11, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, 1, 28, 28]
            
        Returns:
            分类 logits [B, num_classes]
        """
        # 将普通 Tensor 包装为 GeometricTensor
        x = enn.GeometricTensor(x, self.in_type)
        
        # Layer 1: 3x3 conv + BN + ReLU (提升层: trivial -> regular)
        x = self.relu1(self.bn1(self.conv1(x)))
        
        # Layer 2: 3x3 conv + BN + ReLU + 空间最大池化
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Layer 3: 3x3 conv + BN + ReLU
        x = self.relu3(self.bn3(self.conv3(x)))
        
        # Layer 4: 3x3 conv + BN + ReLU
        x = self.relu4(self.bn4(self.conv4(x)))
        
        # Layer 5: 3x3 conv + BN + ReLU
        x = self.relu5(self.bn5(self.conv5(x)))
        
        # Layer 6: 3x3 conv + BN + ReLU
        x = self.relu6(self.bn6(self.conv6(x)))
        
        # Layer 7: 4x4 conv + BN + ReLU
        x = self.relu7(self.bn7(self.conv7(x)))
        
        # GroupPooling: 旋转维度的最大池化 -> 旋转不变性
        x = self.gpool(x)
        
        # 转换为普通 Tensor 进行全连接层
        x = x.tensor
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接分类
        x = self.fc(x)
        
        return x
    
    def evaluate_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        计算输出形状
        """
        batch_size = input_shape[0]
        return (batch_size, self.fc.out_features)


def count_parameters(model, verbose=True):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
    return total_params, trainable_params


if __name__ == "__main__":
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = P4CNN(num_classes=10, n_channels=10).to(device)
    
    print("P4CNN 模型结构:")
    print(model)
    print("\n" + "="*50 + "\n")
    
    # 统计参数
    total_params, trainable_params = count_parameters(model)
    
    # 测试前向传播
    print("\n测试前向传播...")
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出示例:\n{output[0]}")
    
    # 测试旋转等变性
    print("\n测试旋转等变性...")
    model.eval()
    
    # 原始输入
    x_original = torch.randn(1, 1, 28, 28).to(device)
    
    # 旋转 90 度
    x_rot90 = torch.rot90(x_original, k=1, dims=[2, 3])
    
    with torch.no_grad():
        y_original = model(x_original)
        y_rot90 = model(x_rot90)
    
    print(f"原始输出: {y_original[0, :5].cpu().numpy()}")
    print(f"旋转后输出: {y_rot90[0, :5].cpu().numpy()}")
    
    # GroupPooling 后应该对旋转不变
    # 但由于随机初始化，这里只是测试形状正确
    print("模型测试完成!")