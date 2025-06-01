import torch
import torch.nn.functional as F
import lpips  # pip install lpips
import torchvision.transforms as T
import random
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# 假设你已经有 CLIP 图像编码器，如 OpenAI 的 CLIP
# import clip
# model, preprocess = clip.load("ViT-B/32")

# 示例 EOT 图像变换函数：使用高斯滤波
def eot_transforms(img_tensor, num_samples=5, kernel_size=5, sigma_range=(0.5, 1.5)):
    transforms = []
    for _ in range(num_samples):
        sigma = random.uniform(*sigma_range)
        blurred = TF.gaussian_blur(img_tensor, kernel_size=kernel_size, sigma=sigma)
        transforms.append(blurred)
    return torch.cat(transforms, dim=0)

# 对抗攻击损失函数
class AdversarialLoss(torch.nn.Module):
    def __init__(self, clip_model, lpips_model, eot_samples=5):
        super().__init__()
        self.clip_model = clip_model.eval()
        self.lpips_model = lpips_model.eval()
        self.eot_samples = eot_samples

    def forward(self, adv_img, orig_img, target_img):
        # EOT图像变换（高斯模糊）
        adv_eot = eot_transforms(adv_img, num_samples=self.eot_samples)

        # CLIP编码器特征距离（MSE或余弦距离）
        with torch.no_grad():
            adv_feat = self.clip_model.encode_image(adv_eot).float()
            target_feat = self.clip_model.encode_image(target_img.repeat(self.eot_samples, 1, 1, 1)).float()
            clip_loss = F.mse_loss(adv_feat, target_feat)

        # LPIPS感知距离（对抗图与原图之间）
        lpips_loss = self.lpips_model(adv_img, orig_img).mean()

        total_loss = clip_loss + lpips_loss
        return total_loss, clip_loss, lpips_loss
