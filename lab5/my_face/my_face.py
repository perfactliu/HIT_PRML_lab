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


# 主程序
def main():
    print('cuda is ', torch.cuda.is_available())
    model_path = 'stylegan3-t-ffhq-1024x1024.pkl'
    G = load_pretrained_GAN(model_path)
    model, preprocess = load_pretrained_CLIP()
    for param in G.parameters():
        param.requires_grad = False  # 冻结 G 的参数

    for param in model.parameters():
        param.requires_grad = False  # 冻结 model 的参数

    bridge = FeatureToLatent(1024, 512).cuda()
    optimizer = optim.Adam(bridge.parameters(), lr=1e-5)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)
    # num_epoch = 3
    num_step = 200
    # img_path = 'outputs/'

    writer = SummaryWriter('runs')
    my_face = preprocess(Image.open('my_face.jpg')).unsqueeze(0).cuda()
    my_face.requires_grad_()
    features_target = model.encode_image(my_face).cuda()
    features_target = features_target / features_target.norm(dim=-1, keepdim=True)  # 归一化
    transform = transforms.ToTensor()
    my_face_tar = transform(Image.open('my_face.jpg')).unsqueeze(0).cuda()

    # train
    for step in range(num_step):
        # 使用 bridge 生成潜在向量
        generated_latent = bridge(features_target.float())

        c = None  # class labels (not used in this example)
        img_data = G(generated_latent, c)  # NCHW, float32, dynamic range [-1, +1], no truncation

        if step == 0:
            gen_img = save_generated_image(img_data, 'outputs/my_face_first.jpg')
            print('First output of my face saved.')
        elif step == num_step - 1:
            gen_img = save_generated_image(img_data, 'outputs/my_face_last.jpg')
            print('Last output of my face saved.')
        else:
            gen_img = img_data

        # features = model.encode_image(my_face)
        # features = features / features.norm(dim=-1, keepdim=True)  # 归一化
        # features.requires_grad_()
        # # print(features.requires_grad_())

        # 计算损失：目标embedding与生成的embedding之间的欧氏距离
        # print(gen_img.shape)
        # print(my_face.shape)
        loss = F.mse_loss(gen_img, my_face_tar)
        writer.add_scalar('my_face_loss', loss.item(),step)

        # 反向传播并优化
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        # scheduler.step(loss)
        # writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)

        print(step, loss.item())


    torch.save({
        'model_state_dict': bridge.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'feature_to_latent_model.pth')  # 保存文件在当前代码文件的同级目录下
    print('Model saved')


if __name__ == "__main__":
    main()
