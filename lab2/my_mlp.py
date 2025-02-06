import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from torch.utils.tensorboard import SummaryWriter


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is : {device}')

# 定义超参数
input_size = 28 * 28  # MNIST输入大小
hidden_size1 = 128  # 第一个隐层神经元数
hidden_size2 = 128  # 第二个隐层神经元数
num_classes = 10  # 输出类别
num_epochs = 10  # 训练轮次
batch_size = 64  # 批大小
learning_rate = 0.001  # 学习率

# 数据预处理
print('logging dataset...')
transform = transforms.Compose([
    transforms.ToTensor(),  # 输入(28，28)(0-255) 输(28，28)(0-1)
    transforms.Normalize((0.5,), (0.5,))  # 输出(28，28)(-1-1)
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./MNIST', train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root='./MNIST', train=False, transform=transform, download=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# 自定义线性层
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyLinear, self).__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features) * 0.01)  # (out,in)
        self.bias = nn.Parameter(torch.zeros(out_features))  # (out)

    def forward(self, x):
        return x @ self.weights.T + self.bias  # 矩阵乘法加偏置


# 自定义Dropout层
class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super(MyDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = (torch.rand(x.size(0), x.size(1), device=x.device) > self.p).float()
            return x * mask / (1 - self.p)  # 为了保持输出的期望值不变，结果除以 (1 - self.p) 进行归一化。
        return x


# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = MyLinear(input_size, hidden_size1)
        self.dropout1 = MyDropout(0.5)
        self.fc2 = MyLinear(hidden_size1, hidden_size2)
        self.dropout2 = MyDropout(0.5)
        self.fc3 = MyLinear(hidden_size2, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, input_size)  # 将28x28图像展平,即(batch_size,28x28)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.softmax(self.fc3(x))
        return x  # (batch_size,10)


# 初始化模型、损失函数和优化器(和学习率调度器)
model = MLP().to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# todo:学习率模式
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# 训练模型
writer = SummaryWriter('runs/hs1_128_hs2_128_lr_0.001_drop_0.5')  # 日志目录
step = 0
print('training start...')
start = time.time()  # STARTING TIME
for epoch in range(num_epochs):
    model.train()
    batch_num = 0
    for images, labels in train_loader:  # (batch_size,28,28),(batch_size)
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss.item(), step)

        if (batch_num+1) % 100 == 0:
            now = time.time()
            prediction = (now - start)*(60000*num_epochs-epoch*60000-(batch_num+1)*batch_size)//(epoch*60000+(batch_num+1)*batch_size)
            seconds = int((now - start) % 60)
            minutes = int(((now - start) // 60) % 60)
            seconds_pre = int(prediction % 60)
            minutes_pre = int((prediction // 60) % 60)
            print(f'Epoch [{epoch+1}/{num_epochs}],Batch[{batch_num+1}/{60000//batch_size}]:finished  \
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
                    # todo:学习率模式
                    scheduler.step(var_loss)
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('Learning_rate', current_lr, step)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels_v).sum().item()
                accuracy = correct / total
                writer.add_scalar('Accuracy/verification', accuracy, step)
                print(f'Current accuracy of the model on the test images: {100 * correct / total:.2f}%')
        batch_num += 1
        step += 1


# 验证模型
print('evaluation start...')
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
