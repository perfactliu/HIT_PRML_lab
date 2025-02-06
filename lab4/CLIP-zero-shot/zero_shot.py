import os
import torch
import clip
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn, optim
import time
import random
from torch.utils.tensorboard import SummaryWriter

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device is : {device}')

# 加载CLIP模型
model, preprocess = clip.load("RN50", device=device)
model.eval()


# 准备数据集
def load_images(image_folder):
    images = []
    labels = []
    for label, class_name in enumerate(sorted(os.listdir(image_folder))):
        class_folder = os.path.join(image_folder, class_name)
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                    images.append(img)
                    labels.append(label)
    print(f"Finish loading data set. Total figure number:{len(labels)}")
    return torch.cat(images), np.array(labels)


# 加载数据集
print("Loading data set...")
all_images, all_labels = load_images('data/caltech101_test/')  # 数据集路径

# 自动划分训练集（80%）和验证集（20%）
train_images, val_images, train_labels, val_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=26
)

# 准备标签文本
classes = sorted(os.listdir('data/caltech101_test/'))  # 确保这里的路径与上面一致
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)


# 提取特征
def encode_images(images):
    with torch.no_grad():
        features = model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)  # 归一化
    return features


def get_accuracy(similarities, true_labels):
    predictions = similarities.argmax(dim=-1)
    return (predictions.cpu().numpy() == true_labels).mean()


# 分批次验证，一批64个图像，每批结束时计算综合准确率并打印
batch_size = 64
start = time.time()
accuracy = 0
count = 0
# writer = SummaryWriter('runs/seed_26')
for batch_num in range(0, len(val_images), batch_size):
    batch_images = val_images[batch_num:batch_num+batch_size]
    batch_labels = val_labels[batch_num:batch_num+batch_size]

    # 计算图像特征
    val_features = encode_images(batch_images)

    # 计算文本特征
    text_features = model.encode_text(text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 计算相似度
    similarities = val_features @ text_features.T  # (N_train, N_text)
    val_accuracy = get_accuracy(similarities, batch_labels)
    accuracy += val_accuracy
    count += 1
    # writer.add_scalar('Accuracy_42', val_accuracy, count)

    now = time.time()
    prediction = (now - start) * (len(val_images) - (batch_num + 1)) // (batch_num + 1)
    seconds = int((now - start) % 60)
    minutes = int((now - start) // 60)
    seconds_pre = int(prediction % 60)
    minutes_pre = int(prediction // 60)
    print(f'Batch[{count}/{len(val_images) // 64}]:finished  Val Accuracy: {val_accuracy:.4f} \
        time_costed:{minutes}min {seconds}s    time_predicted:{minutes_pre}min {seconds_pre}s')

print(f"Final Val Accuracy: {accuracy/count:.4f} , seed: 26")
