"""
传统 CNN 对照模型
与 P4CNN 相同的空间流程（28→28→pool→14→…→11×11→全连接），用于 Rotated MNIST 对照实验。
前 6 层宽度为 width，第 7 层卷积输出 width_last，便于二维搜索使总参数量与 P4CNN 几乎一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from p4cnn import P4CNN


def _baseline_param_count_analytic(width: int, width_last: int) -> int:
    """与 BaselineCNN 结构一致的参数量（Conv bias=False，BN affine，fc 含 bias）。"""
    w, wl = width, width_last
    n = 9 * w + 2 * w
    for _ in range(5):
        n += 9 * w * w + 2 * w
    n += 16 * w * wl + 2 * wl
    n += wl * 121 * 10 + 10
    return n


def _param_count_module(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class BaselineCNN(nn.Module):
    """
    非等变 7 层卷积 + BN + ReLU；第 7 层输出通道可为 width_last（可与 width 不同以便对齐参数量）。
    """

    def __init__(self, num_classes: int = 10, width: int = 20, width_last: Optional[int] = None):
        super().__init__()
        w = width
        wl = width_last if width_last is not None else w
        self.width = w
        self.width_last = wl

        self.conv1 = nn.Conv2d(1, w, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        self.conv2 = nn.Conv2d(w, w, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(w)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(w, w, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(w)
        self.conv4 = nn.Conv2d(w, w, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(w)
        self.conv5 = nn.Conv2d(w, w, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(w)
        self.conv6 = nn.Conv2d(w, w, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(w)
        self.conv7 = nn.Conv2d(w, wl, kernel_size=4, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(wl)

        self.fc = nn.Linear(wl * 11 * 11, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = x.flatten(1)
        return self.fc(x)


def make_baseline_matched_to_p4cnn(
    num_classes: int = 10,
    n_channels: int = 10,
    search_w_max: int = 256,
) -> Tuple[BaselineCNN, int, int, int, int]:
    """
    构造 BaselineCNN，使总参数量与 P4CNN(num_classes, n_channels) 尽可能接近。

    Returns:
        model: 对照网络
        width: 前 6 层及第 7 层输入侧宽度
        width_last: 第 7 层输出通道（可与 width 不同以便对齐参数量）
        ref_params: 参照 P4CNN 总参数量
        baseline_params: 对照网络总参数量
    """
    ref = P4CNN(num_classes=num_classes, n_channels=n_channels)
    ref_params = _param_count_module(ref)
    del ref

    best_diff = 10**18
    best_pair = (8, 8)

    w_hi = min(search_w_max, max(32, int(ref_params ** 0.5) + 32))
    for w in range(3, w_hi + 1):
        wl_hi = min(search_w_max, max(w, int(ref_params / (w * w + 1)) + 8))
        for wl in range(3, wl_hi + 1):
            n = _baseline_param_count_analytic(w, wl)
            d = abs(n - ref_params)
            if d < best_diff:
                best_diff = d
                best_pair = (w, wl)
                if d == 0:
                    break
        if best_diff == 0:
            break

    w, wl = best_pair
    model = BaselineCNN(num_classes=num_classes, width=w, width_last=wl)
    baseline_params = _param_count_module(model)
    assert baseline_params == _baseline_param_count_analytic(w, wl)
    return model, w, wl, ref_params, baseline_params
