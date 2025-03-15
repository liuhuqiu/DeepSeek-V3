import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib.pyplot as plt

block_size:int = 128

# NLP例子  
batch, sentence_length, embedding_dim = 5, 3, 4
layer_norm = nn.LayerNorm(embedding_dim)  # 在embedding维度上归一化
x1 = torch.cat((torch.ones(2), torch.zeros(2)), dim=0)
x2 = layer_norm(x1)
layer_norm = nn.RMSNorm(embedding_dim)  # 在embedding维度上归一化
x = layer_norm(x1)
print(f"layer_norm:{layer_norm}, x1:{x1}, x2:{x2}, x:{x}")
# 输入 1 1 0 0 ==》 归一化后为 1 1 -1 -1

# 2维度归一化 
N, C = 4, 4
x3 = 100 * torch.rand(N, C, dtype=torch.float32)
layer_norm = nn.LayerNorm([N, C])  # 在通道和空间维度上归一化,全部初始化掉
x4 = layer_norm(x3)
layer_norm = nn.RMSNorm([N, C])  # 在embedding维度上归一化
x = layer_norm(x3)
print(f"layer_norm:{layer_norm}, x3:{x3}, x4:{x4}, x:{x}")

# 使用 matplotlib 绘制点状图
plt.figure(figsize=(18, 6))

# 子图 1: x4 (LayerNorm 后的结果)
plt.subplot(1, 3, 1)
for i in range(C):
    plt.scatter(x3[i,:].detach().cpu().numpy(), x3[:, i].detach().cpu().numpy(), label=f'dim {i}')
plt.title("Original Data")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

# 子图2
plt.subplot(1, 3, 2)
for i in range(C):
    plt.scatter(x4[i, :].detach().cpu().numpy(), x4[:, i].detach().cpu().numpy(), label=f'dim {i}', marker='x')
plt.title("After LayerNorm")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

# 子图 2: x (RMSNorm 后的结果)
plt.subplot(1, 3, 3)
for i in range(C):
    plt.scatter(x[i, :].detach().cpu().numpy(), x[:, i].detach().cpu().numpy(), label=f'dim {i}', marker='^')
plt.title("After RMSNorm")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()

# 示例1：在embedding维度上归一化
layer_norm = nn.LayerNorm(normalized_shape=C)
# 这会对最后一个维度(256)进行归一化
x5 = torch.rand(N, C, dtype=torch.bfloat16)
x6 = layer_norm(x5)
print(f"layer_norm:{layer_norm}, x5:{x5}, x6:{x6}")

# 示例2：在序列和embedding维度上归一化
layer_norm = nn.LayerNorm(normalized_shape=[50, 256])
# 这会对最后两个维度一起进行归一化
# 每个样本的所有词向量一起计算均值和方差
print(f"layer_norm:{layer_norm}")