import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# 自定义的转换，确保灰度图像转换为 3 通道的 RGB 图像
def to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')  # 如果图像不是RGB格式，转换为RGB
    return image


# # 定义预处理
# transform = transforms.Compose([
#     transforms.Lambda(to_rgb),  # 确保所有图像都有3个通道
#     transforms.Resize((224, 224)),  # 将图像调整为 224x224
#     transforms.ToTensor(),          # 将图像转换为张量
# ])
#
# # 加载 Caltech101 数据集
# train_dataset = datasets.Caltech101(root='./data', download=False, transform=transform)
#
# # 创建 DataLoader
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
#
# for images, labels in train_loader:
#     print(labels)

# # 获取一个批次的数据
# images, labels = next(iter(train_loader))
#
# # 打印批次数据的形状
# print(f"Images shape: {images.shape}")  # 应该是 torch.Size([64, 3, 224, 224])
# print(f"Labels shape: {labels.shape}")  # torch.Size([64])
# print(f"Labels: {labels}")

import torch
from torch.utils.data import random_split, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 定义预处理
transform = transforms.Compose([
    transforms.Lambda(to_rgb),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载 Caltech101 数据集
dataset = datasets.Caltech101(root='./data', download=False, transform=transform)

# 计算数据集中训练集和测试集的大小
train_size = int(0.8 * len(dataset))  # 80% 的数据用作训练集
test_size = len(dataset) - train_size  # 剩下的 20% 用作测试集

# 使用 random_split 划分数据集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 打印数据集大小
print(f"Training set size: {len(train_dataset)}")
print(f"Testing set size: {len(test_dataset)}")
# 获取一个批次的数据
images, labels = next(iter(train_loader))

# 打印批次数据的形状
print(f"Images shape: {images.shape}")  # 应该是 torch.Size([64, 3, 224, 224])
print(f"Labels shape: {labels.shape}")  # torch.Size([64])
print(f"Labels: {labels}")
