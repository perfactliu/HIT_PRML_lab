import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from transformers import CLIPTextModel, CLIPTokenizer
from loralib import LoRALayer
import os
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict


# 定义Dataset
class ImageTextDataset(Dataset):
    def __init__(self, img_dir, text, tokenizer, transform=None):
        self.img_dir = img_dir
        self.text = text
        self.img_files = sorted(os.listdir(img_dir))  # 图片按照数字命名
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # 文本提示 "a dog"
        text_input = self.tokenizer(self.text, return_tensors="pt", padding=True, truncation=True)

        return image, text_input.input_ids.squeeze()

# 加载预训练模型
print("loading model...")
noise_scheduler = DDPMScheduler.from_pretrained("stable-diffusion-v1-5", subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(
    "stable-diffusion-v1-5", subfolder="tokenizer"
)
text_encoder = CLIPTextModel.from_pretrained(
    "stable-diffusion-v1-5", subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(
    "stable-diffusion-v1-5", subfolder="vae"
)
unet = UNet2DConditionModel.from_pretrained(
    "stable-diffusion-v1-5", subfolder="unet"
)
# 冻结VAE和文本编码器
for param in vae.parameters():
    param.requires_grad = False
for param in text_encoder.parameters():
    param.requires_grad = False

# 使用LoRA微调UNet
for param in unet.parameters():
    param.requires_grad_(False)

unet_lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)
unet.add_adapter(unet_lora_config)
lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

# 使用transformer處理圖像
train_transforms = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256) ,
            transforms.RandomHorizontalFlip() ,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

# 数据加载器
print("loading dataset...")
dataset = ImageTextDataset(img_dir="./target", text="a dog", tokenizer=tokenizer, transform=train_transforms)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 优化器
optimizer_cls = torch.optim.AdamW
optimizer = optimizer_cls(
    lora_layers,
    lr=1e-4,
)
# print(f'number of training parameters of sft(lora): {sum(p.numel() for p in unet.parameters() if p.requires_grad)}')

# 微调循环
print("start training...")
writer = SummaryWriter('runs')
num_epochs = 100
for epoch in range(num_epochs):
    loss_written=0.0
    unet.train()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for image, text_ids in progress_bar:
        latents = vae.encode(image.to(dtype=torch.float32)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # 获取文本嵌入
        encoder_hidden_states = text_encoder(text_ids, return_dict=False)[0]
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        loss_written = loss
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新进度条
        progress_bar.set_postfix(loss=loss.item())

    writer.add_scalar('Loss/train', loss_written.item(), epoch)

# 保存微调后的模型
print("model saved.")
unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
StableDiffusionPipeline.save_lora_weights(
    save_directory="./fine_tuned_lora_unet",
    unet_lora_layers=unet_lora_state_dict,
    safe_serialization=True,
)

