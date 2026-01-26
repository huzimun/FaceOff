import torch
from PIL import Image
import os


def generate_random_noise_image(
    height: int,
    width: int,
    channels: int = 3,
    batch_size: int = 1,
    noise_type: str = "uniform",
    device: str = "cpu"
):
    """
    Generate random noise images.

    Returns:
        torch.Tensor: (B, C, H, W), float32
    """
    if noise_type == "uniform":
        # Uniform noise in [0, 1]
        noise = torch.rand(batch_size, channels, height, width, device=device)
    elif noise_type == "gaussian":
        # Gaussian noise with mean=0, std=1
        noise = torch.randn(batch_size, channels, height, width, device=device)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    return noise


def save_tensor_as_image(
    tensor: torch.Tensor,
    save_path: str,
    index: int = 0
):
    """
    Save a tensor image to disk.

    Args:
        tensor (torch.Tensor): (B, C, H, W), float
        save_path (str): output file path (.png / .jpg)
        index (int): which image in batch to save
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = tensor[index].detach().cpu()

    # Normalize to [0, 1] for visualization
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Convert to uint8
    img = (img * 255).clamp(0, 255).byte()

    # (C, H, W) -> (H, W, C)
    img = img.permute(1, 2, 0).numpy()

    Image.fromarray(img).save(save_path)


# ===== Example usage =====
noise = generate_random_noise_image(
    height=256,
    width=256,
    channels=3,
    batch_size=1,
    noise_type="gaussian"
)

save_tensor_as_image(noise, "/home/humw/Codes/FaceOff/target_images/noise/random_noise.png")
