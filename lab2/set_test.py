from torchvision import datasets, transforms
import torch

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载训练集和测试集
train_dataset = datasets.MNIST(root='./mnist', train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root='./mnist', train=False, transform=transform, download=False)

print(f"训练集大小: {len(train_dataset)}")  # 应该输出 60000
print(f"测试集大小: {len(test_dataset)}")  # 应该输出 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
