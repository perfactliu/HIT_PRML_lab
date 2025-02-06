import sys
import os

stylegan_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stylegan3-main'))
clip_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CLIP'))
sys.path.append(stylegan_path)
sys.path.append(clip_path)
import torch
from torch import nn
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import shutil
import dnnlib
import legacy
import clip
from torch.optim import lr_scheduler

# 定义全连接转换网络
class FeatureToLatent(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(FeatureToLatent, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# 加载预训练的StyleGAN2模型
def load_pretrained_GAN(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with dnnlib.util.open_url(model_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # G_ema 是生成器
    # G.eval()
    return G


# 加载预训练的clip模型
def load_pretrained_CLIP():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load("RN50", device=device)
    # model.eval()
    return model, preprocess


# 可视化生成的图像
def save_generated_image(generated_img, output_path):
    # 先进行 detach 和 CPU 转换，以便在保存图像时不影响计算图
    img_to_save = generated_img.detach().cpu().numpy()  # 将需要保存的图像转为 NumPy 数组
    # 检查并调整数组形状，如果是 (1, 1, H, W) 或 (1, C, H, W)，去掉多余的维度
    if img_to_save.ndim == 4:  # 可能是 (1, 1, H, W) 或 (1, C, H, W)
        img_to_save = img_to_save.squeeze(0)  # 移除第一维，形状变成 (1, H, W) 或 (C, H, W)
        if img_to_save.ndim == 3:  # 处理 (C, H, W) 形状
            img_to_save = np.transpose(img_to_save, (1, 2, 0))  # 转为 (H, W, C)
    # 确保生成的图像在 [0, 255] 范围内
    img_to_save = (img_to_save * 255).clip(0, 255).astype(np.uint8)
    # 创建 PIL 图像并保存
    img = Image.fromarray(img_to_save)
    img.save(output_path)
    # 返回原始张量（未被 detach），以便继续参与计算图
    return generated_img

def main():
    print('cuda is ', torch.cuda.is_available())
    model_path = 'stylegan3-t-ffhq-1024x1024.pkl'
    G = load_pretrained_GAN(model_path)
    model, preprocess = load_pretrained_CLIP()
    for param in G.parameters():
        param.requires_grad = False  # 冻结 G 的参数

    for param in model.parameters():
        param.requires_grad = False  # 冻结 model 的参数

    bridge = FeatureToLatent(1024, 512)
    checkpoint = torch.load('feature_to_latent_model.pth')
    bridge.load_state_dict(checkpoint['model_state_dict'])
    bridge.eval().cuda()

    my_face = preprocess(Image.open('my_face.jpg')).unsqueeze(0).cuda()
    features = model.encode_image(my_face).cuda()
    features = features / features.norm(dim=-1, keepdim=True)  # 归一化
    generated_latent = bridge(features.float())
    generated_latent.cuda()

    w = G.mapping(generated_latent,None)  # 映射到风格空间
    offset_1_4 = torch.randn(1, 4, 512).cuda()  # 对应1-4层的偏移量
    offset_4_8 = torch.randn(1, 4, 512).cuda()  # 对应4-8层的偏移量
    offset_8_12 = torch.randn(1, 4, 512).cuda()  # 对应8-12层的偏移量
    offset_12_16 = torch.randn(1, 4, 512).cuda()  # 对应12-16层的偏移量

    num_step = 50

    # train
    for step in range(num_step):
        w[:, 0:4, :] += offset_1_4*0.03  # 对1-4层添加偏移量
        # w[:, 4:8, :] += offset_4_8*0.03  # 对4-8层添加偏移量
        # w[:, 8:12, :] += offset_8_12*0.03  # 对8-12层添加偏移量
        # w[:, 12:16, :] += offset_12_16*0.03  # 对12-16层添加偏移量
        img_data = G.synthesis(w)
        save_generated_image(img_data, '12_16/'+str(step)+'.jpg')
        print(f'figure {step} saved.')


if __name__ == "__main__":
    main()