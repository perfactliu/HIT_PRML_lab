import math
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


class AutoPrompt(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16  # 嵌入数量
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # 初始化
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        if torch.cuda.is_available():
            ctx_vectors = ctx_vectors.cuda()
        nn.init.normal_(ctx_vectors, std=0.02)  # 方差为0.02的高斯分布
        prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)  # 训练时自动更新

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        if torch.cuda.is_available():
            tokenized_prompts = tokenized_prompts.cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1+n_ctx:, :]
        self.tokenized_prompts = tokenized_prompts
        self.n_cls = n_cls
        self.n_ctx = n_ctx

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = AutoPrompt(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale*image_features @ text_features.T

        return logits


# 准备数据集，返回值：img:(N,C,H,W)  labels:(N:102*img_num)
def load_images(image_folder):
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    for label, class_name in enumerate(sorted(os.listdir(image_folder))):
        images = []
        labels = []
        class_folder = os.path.join(image_folder, class_name)
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)  # data/caltech101_test/***(class)/image_xxxx.jpg
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                    images.append(img)
                    labels.append(label)
        train_images_class, val_images_class, train_labels_class, val_labels_class = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )
        train_images.extend(train_images_class)
        train_labels.extend(train_labels_class)
        val_images.extend(val_images_class)
        val_labels.extend(val_labels_class)
    print(f"Finish loading data set. Train figure number:{len(train_labels)}. Verification figure number:{len(val_labels)}")
    return torch.cat(train_images), np.array(train_labels), torch.cat(val_images), np.array(val_labels)


# 从训练集中选择指定数量的样本的函数
def select_shots(train_images, train_labels, num_shots):
    selected_images = []
    selected_labels = []
    for label in set(train_labels):
        class_indices = [i for i, lbl in enumerate(train_labels) if lbl == label]
        sampled_indices = random.sample(class_indices, num_shots)

        selected_images.extend(train_images[sampled_indices])
        selected_labels.extend([label] * len(sampled_indices))
    print(f"Class of train set: {selected_labels}")
    # 创建一个索引列表并打乱顺序
    indices = list(range(len(selected_labels)))
    random.shuffle(indices)
    # 根据打乱的索引重新排列两个列表
    shuffled_images = [selected_images[i] for i in indices]
    shuffled_labels = [selected_labels[i] for i in indices]
    return torch.stack(shuffled_images), np.array(shuffled_labels)


# 准确率计算
def get_accuracy(similarities, true_labels):
    predictions = similarities.argmax(dim=-1)
    return (predictions.cpu().numpy() == true_labels).mean()


# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device is : {device}')
# 准备标签文本
classnames = sorted(os.listdir('data/caltech101_test/'))

# 加载CLIP模型与CoOp模型
print("Loading CoOp model...")
download_path = 'model_saver'  # 自定义的缓存路径
# 清空缓存文件夹（可以选择性删除缓存文件夹内容）
if os.path.exists(download_path):
    import shutil
    shutil.rmtree(download_path)
# 重新加载 CLIP 模型，强制下载
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("RN50", device=device, download_root=download_path)
model = CustomCLIP(classnames, clip_model)
model.train()  # 设置模型为训练模式

# 加载数据集
print("Loading data set...")
train_images, train_labels, val_images, val_labels = load_images('data/caltech101_test/')
# 设定每个类别的 shot 数
num_shots = 1
print(f"Loading {num_shots} shot(s) training data set.")
train_images, train_labels = select_shots(train_images, train_labels, num_shots)

# 训练设置
num_epochs = 50
batch_size = 64
initial_learning_rate = 0.002
warmup_learning_rate = 1e-5
optimizer = optim.SGD(model.prompt_learner.parameters(), lr=initial_learning_rate)

# 使用余弦退火调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# 训练过程
print("Training start.")
start = time.time()
# writer = SummaryWriter('runs/1shot/seed_42')
step = 0
for epoch in range(num_epochs):
    model.train()
    # Warmup阶段
    if epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_learning_rate

    for batch_num in range(0, len(train_images), batch_size):
        step += 1
        batch_images = train_images[batch_num:batch_num + batch_size]
        batch_labels = train_labels[batch_num:batch_num + batch_size]
        # 计算相似度
        similarities = model(batch_images)  # (N_train, N_text)

        # 计算损失
        loss = nn.CrossEntropyLoss()(similarities, torch.tensor(batch_labels, dtype=torch.long).to(device))
        # writer.add_scalar('Loss', loss.item(), step)
        print(f"Loss: {loss.item():.4f}")

        # 更新模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新学习率
        scheduler.step()
    # 每个epoch结束时进行验证
    now = time.time()
    prediction = (now - start) * (num_epochs - epoch - 1) // (epoch + 1)
    seconds = int((now - start) % 60)
    minutes = int((now - start) // 60)
    seconds_pre = int(prediction % 60)
    minutes_pre = int(prediction // 60)
    print(f'Epoch [{epoch + 1}/{num_epochs}]: finished , time cost: {minutes}min {seconds}s , predicted time: {minutes_pre}min {seconds_pre}s')
    print('Verification start.')
    model.eval()
    num_samples = 64  # 希望验证的样本数量
    # 随机选择验证样本索引
    sample_indices = random.sample(range(len(val_images)), num_samples)
    sampled_val_images = val_images[sample_indices]
    sampled_val_labels = val_labels[sample_indices]
    with torch.no_grad():  # 禁用梯度计算
        val_similarities = model(sampled_val_images)
    # 计算验证准确率
    val_accuracy = get_accuracy(val_similarities, sampled_val_labels)
    # writer.add_scalar('Accuracy', val_accuracy, epoch+1)
    print(f"Val Accuracy: {val_accuracy:.4f}")

# 最终验证
model.eval()
with torch.no_grad():  # 禁用梯度计算
    val_similarities = model(val_images)
final_accuracy = get_accuracy(val_similarities, val_labels)
print(f"Final Val Accuracy: {final_accuracy:.4f}")
