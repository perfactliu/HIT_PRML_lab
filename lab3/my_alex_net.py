import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split, DataLoader
import time
from torch.utils.tensorboard import SummaryWriter


# 自定义的转换，确保灰度图像转换为 3 通道的 RGB 图像
def to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')  # 如果图像不是RGB格式，转换为RGB
    return image


# 定义 AlexNet 结构
class AlexNet(nn.Module):
    def __init__(self, num_classes=102):  # Caltech101有102个类
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(  # (3,224,224)
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # Conv Layer 1  (96,54,54)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Pooling Layer 1  (96,26,26)
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # Conv Layer 2  (256,26,26)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Pooling Layer 2  (256,12,12)
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # Conv Layer 3  (384,12,12)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # Conv Layer 4  (384,12,12)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # Conv Layer 5  (256,12,12)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # Pooling Layer 3  (256,5,5)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # 256 * 6 * 6 是池化后的特征图尺寸
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平成 1D 向量
        x = self.classifier(x)
        return x


print('logging dataset...')
# 定义数据预处理流程
transform = transforms.Compose([
    transforms.Lambda(to_rgb),  # 确保所有图像都有3个通道
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(20),  # 随机旋转 [-20, 20] 度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 加载 Caltech101 数据集
dataset = datasets.Caltech101(root='./data', download=False, transform=transform)
# 计算数据集中训练集和测试集的大小
train_size = int(0.8 * len(dataset))  # 80% 的数据用作训练集
test_size = len(dataset) - train_size  # 剩下的 20% 用作测试集
# 使用 random_split 划分数据集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化 AlexNet 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is : {device}')
model = AlexNet(num_classes=102).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    start = time.time()
    step = 0
    writer = SummaryWriter('runs/origin_no_drop')
    for epoch in range(num_epochs):
        batch_num = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算损失和准确率
            writer.add_scalar('Loss/train', loss.item(), step)
            if (batch_num + 1) % 10 == 0:
                now = time.time()
                prediction = (now - start) * (6941 * num_epochs - epoch * 6941 - (batch_num + 1) * 64) // (
                            epoch * 6941 + (batch_num + 1) * 64)
                seconds = int((now - start) % 60)
                minutes = int((now - start) // 60)
                seconds_pre = int(prediction % 60)
                minutes_pre = int(prediction // 60)
                print(f'Epoch [{epoch + 1}/{num_epochs}],Batch[{batch_num + 1}/{6941 // 64}]:finished  \
                time_costed:{minutes}min {seconds}s    time_predicted:{minutes_pre}min {seconds_pre}s')

                # 验证集测试
                print('verification start...')
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images_v, labels_v in test_loader:
                        images_v, labels_v = images_v.to(device), labels_v.to(device)
                        outputs = model(images_v)
                        var_loss = criterion(outputs, labels_v)
                        writer.add_scalar('Loss/verification', var_loss, step)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels_v).sum().item()
                    accuracy = correct / total
                    writer.add_scalar('Accuracy/verification', accuracy, step)
                    print(f'Current accuracy of the model on the test images: {100 * correct / total:.2f}%')
            batch_num += 1
            step += 1


# 测试函数
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')


# 开始训练与测试
print('training start...')
train_model(model, train_loader, criterion, optimizer, num_epochs=20)
print('evaluation start...')
test_model(model, test_loader)
