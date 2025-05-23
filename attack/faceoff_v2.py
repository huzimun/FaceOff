import argparse
import copy
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path

import datasets
import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import torchvision
from transformers import AutoTokenizer, PretrainedConfig
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
import numpy as np
import pdb
import random
import sys

logger = get_logger(__name__)

import torchvision.transforms as transforms

import torch
import torch.nn as nn
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection


def save_image(save_dir, input_dir, perturbed_data):
    os.makedirs(save_dir, exist_ok=True)
    noised_imgs = perturbed_data.detach()
    img_names = [
        str(instance_path).split("/")[-1]
        for instance_path in list(Path(input_dir).iterdir())
    ]
    for img_pixel, img_name in zip(noised_imgs, img_names):
        save_path = os.path.join(save_dir, img_name)
        Image.fromarray(
            img_pixel.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        ).save(save_path)
    print("save images to {}".format(save_dir))
    
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--eot",
        type=int,
        default=0,
        required=True,
        help="1 use eot, 0 not use eot",
    )
    parser.add_argument(
        "--model_types", 
        type=str, 
        default="vae15", 
        help="model types string split with ;")
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0", 
        help="gpu id")
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training.")
    parser.add_argument(
        "--target",
        type=str,
        default="yingbu",
        required=True,
        help="yingbu, mist, non-target",
    )
    parser.add_argument(
        "--distance_choice",
        type=str,
        default="mse",
        required=True,
        help="mse or cosine similarity",
    )
    parser.add_argument(
        "--target_image_path",
        default=None,
        help="target image for attacking",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--instance_data_dir_for_adversarial",
        type=str,
        default=None,
        required=True,
        help="A folder containing the images to add adversarial noise",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=20,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--max_adv_train_steps",
        type=int,
        default=10,
        help="Total number of sub-steps to train adversarial noise.",
    )
    parser.add_argument(
        "--pgd_alpha",
        type=float,
        default=1.0 / 255,
        help="The step size for pgd.",
    )
    parser.add_argument(
        "--pgd_eps",
        type=float,
        default=0.05,
        help="The noise budget for pgd.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def load_data(args, data_dir="", size=512, center_crop=True) -> torch.Tensor:
    if args.eot == 0:
        image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        if Path(data_dir).is_file():
            image = Image.open(data_dir).convert("RGB")
            images = [image_transforms(image) for _ in range(4)]
        else:
            images = [image_transforms(Image.open(i).convert("RGB")) for i in list(Path(data_dir).iterdir())]

        images = torch.stack(images)
    else:
        def image_to_numpy(image):
            return np.array(image).astype(np.uint8)
        # more robust loading to avoid loaing non-image files
        images = [] 
        if Path(data_dir).is_file():  
            image = Image.open(data_dir).convert("RGB")
            images.extend([image_to_numpy(image) for _ in range(4)])
        else:
            for i in list(Path(data_dir).iterdir()):
                if not i.suffix in [".jpg", ".png", ".jpeg"]:
                    continue
                else:
                    images.append(image_to_numpy(Image.open(i).convert("RGB")))
        images = [Image.fromarray(i).resize((size, size), 2) for i in images]
        images = np.stack(images)
        # from B x H x W x C to B x C x H x W
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        assert images.shape[-1] == images.shape[-2]
    return images

def pgd_attack(
    args,
    torch_dtype,
    model_dict,
    perturbed_images: torch.Tensor,
    original_images: torch.Tensor,
    target_images: torch.Tensor,
    num_steps: int,
    trans_224,
    trans_336,
    trans_512,
):
    """Return new perturbed data"""
    device = torch.device(args.device)
    perturbed_images = perturbed_images.detach().clone().to(dtype=torch_dtype).to(device)
    # perturbed_images.requires_grad_(True)
    original_images = original_images.requires_grad_(False).to(dtype=torch_dtype).to(device)

    if args.target == "non-target": # 无目标
        # 加入随机扰动
        perturbed_images = (perturbed_images + (torch.rand(*perturbed_images.shape)*2*args.pgd_eps-args.pgd_eps).to(torch_dtype).to(device))
    else:
        target_images = target_images.requires_grad_(False).to(dtype=torch_dtype).to(device)
    
    model_types = list()
    # 遍历模型字典，将每个模型放到gpu上
    for model_type in model_dict.keys():
        model_dict[model_type].eval()
        model_dict[model_type].to(device)
        model_types.append(model_type)
        
    # 获取原始图像和目标图像的编码
    target_embeds_dict = {}
    original_embeds_dict = {}
    for model_type in model_dict.keys():
        if "vae" in model_type:
            tran_original_data_512 = trans_512(original_images).to(dtype=torch_dtype)
            original_image_embeds = model_dict[model_type].encode(tran_original_data_512).latent_dist.sample() * model_dict[model_type].config.scaling_factor
            if args.target != "non-target":
                tran_target_data_512 = trans_512(target_images).to(dtype=torch_dtype)
                target_image_embeds = model_dict[model_type].encode(tran_target_data_512).latent_dist.sample() * model_dict[model_type].config.scaling_factor
        elif "ipadapter" == model_type:
            tran_original_data_224 = trans_224(original_images).to(dtype=torch_dtype)
            original_image_embeds = model_dict[model_type](tran_original_data_224, output_hidden_states=True).hidden_states[-2]
            if args.target != "non-target":
                tran_target_data_224 = trans_224(target_images).to(dtype=torch_dtype)
                target_image_embeds = model_dict[model_type](tran_target_data_224, output_hidden_states=True).hidden_states[-2]
        elif "photomaker" == model_type:
            tran_original_data_224 = trans_224(original_images).to(dtype=torch_dtype)
            original_image_embeds = model_dict[model_type](tran_original_data_224)
            if args.target != "non-target":
                tran_target_data_224 = trans_224(target_images).to(dtype=torch_dtype)
                target_image_embeds = model_dict[model_type](tran_target_data_224)
        else:
            raise NotImplementedError
        if args.target != "non-target":
            target_embeds_dict[model_type] = target_image_embeds
        original_embeds_dict[model_type] = original_image_embeds
    
    pgd_loss_list = list() # 保存损失函数字典
    for step in range(num_steps): # 6
        perturbed_images.requires_grad = True
        # 获取对抗图像的编码
        perturbed_embeds_dict = {}
        loss_dict = {}
        grad_dict = {}
        for model_type in model_dict.keys():
            if "vae" in model_type:
                tran_perturbed_data_512 = trans_512(perturbed_images).to(dtype=torch_dtype)
                perturbed_image_embeds = model_dict[model_type].encode(tran_perturbed_data_512).latent_dist.sample() * model_dict[model_type].config.scaling_factor
            elif "ipadapter" == model_type:
                tran_perturbed_data_224 = trans_224(perturbed_images).to(dtype=torch_dtype)
                perturbed_image_embeds = model_dict[model_type](tran_perturbed_data_224, output_hidden_states=True).hidden_states[-2]
            elif "photomaker" == model_type:
                tran_perturbed_data_224 = trans_224(perturbed_images).to(dtype=torch_dtype)
                perturbed_image_embeds = model_dict[model_type](tran_perturbed_data_224)
            else:
                raise NotImplementedError
            perturbed_embeds_dict[model_type] = perturbed_image_embeds
            model_dict[model_type].zero_grad()
            if args.target == "non-target":
                if args.distance_choice == "mse": # 和原始编码MSE距离越大越好，取负，越小越好
                    loss = - F.mse_loss(original_embeds_dict[model_type], perturbed_embeds_dict[model_type], reduction="mean")
                elif args.distance_choice == "cosine": # 和原始编码余弦相似度越小越好
                    loss = F.cosine_similarity(original_embeds_dict[model_type], perturbed_embeds_dict[model_type], -1).mean()
                else: # mix
                    if "vae" in model_type:
                        loss = - F.mse_loss(original_embeds_dict[model_type], perturbed_embeds_dict[model_type], reduction="mean")
                    else:
                        loss = F.cosine_similarity(original_embeds_dict[model_type], perturbed_embeds_dict[model_type], -1).mean()
            else: # 最小化编码器目标损失函数
                if args.distance_choice == "mse":
                    loss = F.mse_loss(target_embeds_dict[model_type], perturbed_embeds_dict[model_type], reduction="mean")
                elif args.distance_choice == "cosine":
                    loss = - F.cosine_similarity(target_embeds_dict[model_type], perturbed_embeds_dict[model_type], -1).mean()
                else: # mix
                    if "vae" in model_type:
                        loss = F.mse_loss(target_embeds_dict[model_type], perturbed_embeds_dict[model_type], reduction="mean")
                    else:
                        loss = - F.cosine_similarity(target_embeds_dict[model_type], perturbed_embeds_dict[model_type], -1).mean()
            loss_dict[model_type] = loss
            grad = torch.autograd.grad(loss, perturbed_images, retain_graph=True, create_graph=False)[0]
            grad_dict[model_type] = grad
        weighted_loss = 0.0
        sum_loss = 0.0

        alphas = [10, 1, 1]
        for tmp_idx, model_type in enumerate(model_types):
            sum_loss += alphas[tmp_idx] * loss_dict[model_type]
        weighted_loss = sum_loss

        loss_dict["sum_loss"] = sum_loss
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key].item()
        print("Step: {}, loss_dict: {}".format(step, loss_dict))
        pgd_loss_list.append(loss_dict) 
        # import pdb; pdb.set_trace()
        weighted_grad = torch.autograd.grad(weighted_loss, perturbed_images)[0]
        adv_perturbed_data = perturbed_images - args.pgd_alpha * weighted_grad.sign() # Minimize the target loss, so it is a reduction
        et = torch.clamp(adv_perturbed_data - original_images, min=-args.pgd_eps, max=+args.pgd_eps)
        perturbed_images = torch.clamp(original_images + et, min=torch.min(original_images), max=torch.max(original_images)).detach().clone()
    return perturbed_images, pgd_loss_list

def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if args.mixed_precision == "fp32":
        torch_dtype = torch.float32
    elif args.mixed_precision == "fp16":
        torch_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    model_type_list = args.model_types.split("-")
    model_dict = {}
    print("model_type_list: ", model_type_list)
    for model_type in model_type_list:
        if model_type == "clip":
            ipadapter_path = "/data1/xxxx/Pretrains/IP-Adapter/models/image_encoder"
            model = CLIPVisionModelWithProjection.from_pretrained(ipadapter_path).to(dtype=torch_dtype).eval().requires_grad_(False)
        elif model_type == "vggface":
            model = PhotoMakerIDEncoder()
            state_dict = torch.load("/data1/xxxx/Pretrains/photomaker-v1.bin", map_location="cpu")
            model.load_state_dict(state_dict['id_encoder'], strict=True)
            model.to(dtype=torch_dtype).eval().requires_grad_(False)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        model_dict[model_type] = model
    # pdb.set_trace()

    perturbed_data = load_data(
        args,
        data_dir=args.instance_data_dir_for_adversarial,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    if args.target == "non-target":
        target_data = None
    else:
        if (args.target == "max-mask") or (args.target == "min-mask") or (args.target == "random-mask"):
            
            person_id = args.output_dir.split('/')[-1]
            
            with open(args.id_map_path, 'r') as f:
                id_map = json.load(f)
            target_id = id_map[person_id]
            target_image_path = os.path.join(args.target_image_path, target_id)
        if args.target == "face":
            
            person_id = args.output_dir.split('/')[-1]
            
            with open(args.id_map_path, 'r') as f:
                id_map = json.load(f)
            target_id = id_map[person_id]
            target_image_path = os.path.join(args.target_image_path, target_id, "set_B")
        else:
            target_image_path = args.target_image_path
        target_data = load_data(
            args,
            data_dir=target_image_path,
            size=args.resolution,
            center_crop=args.center_crop,
        )
    original_data = perturbed_data.clone()
    original_data.requires_grad_(False)
    
    resample_interpolation = transforms.InterpolationMode.BILINEAR
    if args.eot == 0:
        trans_224 = [
            transforms.Resize(224, interpolation=resample_interpolation),
            transforms.CenterCrop(224) if args.center_crop else transforms.RandomCrop(224),
        ]
        trans_224 = transforms.Compose(trans_224)
        
        trans_336 = [
            transforms.Resize(336, interpolation=resample_interpolation),
            transforms.CenterCrop(336) if args.center_crop else transforms.RandomCrop(336),
        ]
        trans_336 = transforms.Compose(trans_336)
        
        trans_512 = [
            transforms.Resize(512, interpolation=resample_interpolation),
            transforms.CenterCrop(512) if args.center_crop else transforms.RandomCrop(512),
        ]
        trans_512 = transforms.Compose(trans_512)
    else:
        train_aug_224 = [
            transforms.Resize(224, interpolation=resample_interpolation),
            transforms.CenterCrop(224) if args.center_crop else transforms.RandomCrop(224),
        ]

        tensorize_and_normalize = [
            transforms.Normalize([0.5*255]*3,[0.5*255]*3),
        ]
        trans_224 = train_aug_224 + tensorize_and_normalize
        trans_224 = transforms.Compose(trans_224)
        print("all_trans:{}".format(trans_224))
        
        train_aug_336 = [
            transforms.Resize(336, interpolation=resample_interpolation),
            transforms.CenterCrop(336) if args.center_crop else transforms.RandomCrop(336),
        ]

        trans_336 = train_aug_336 + tensorize_and_normalize
        trans_336 = transforms.Compose(trans_336)
        print("all_trans:{}".format(trans_336))
        
        train_aug_512 = [
            transforms.Resize(512, interpolation=resample_interpolation),
            transforms.CenterCrop(512) if args.center_crop else transforms.RandomCrop(512),
        ]
        trans_512 = train_aug_512 + tensorize_and_normalize
        trans_512 = transforms.Compose(trans_512)
        print("all_trans:{}".format(trans_512))
        
        args.pgd_eps = 16.0
        args.pgd_alpha = 16/10 # The default is 1/10 of the threshold
        
    pgd_loss_list = []
    for i in range(args.max_train_steps):
        perturbed_data, tmp_pgd_loss_list  = pgd_attack(
            args,
            torch_dtype,
            model_dict=model_dict,
            perturbed_images=perturbed_data,
            original_images=original_data,
            target_images=target_data,
            num_steps=args.max_adv_train_steps,
            trans_224=trans_224,
            trans_336=trans_336,
            trans_512=trans_512,
        )
        pgd_loss_list.extend(tmp_pgd_loss_list)

    save_folder = args.output_dir
    os.makedirs(save_folder, exist_ok=True)
    noised_imgs = perturbed_data.detach()
    img_names = [
        str(instance_path).split("/")[-1]
        for instance_path in list(Path(args.instance_data_dir_for_adversarial).iterdir())
    ]
    if args.eot == 0:
        for img_pixel, img_name in zip(noised_imgs, img_names):
            save_path = os.path.join(save_folder, img_name)
            Image.fromarray(
                (img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            ).save(save_path)
    else:
        save_image(save_folder, args.instance_data_dir_for_adversarial, noised_imgs)
    print(f"Saved noise images to {save_folder}")
    
    # Save PGD attack loss list
    person_id = args.output_dir.split('/')[-1]
    exp_name = args.output_dir.split('/')[-2]
    config_scripts_logs_path = "/data1/xxxx/Codes/TED/outputs/config_scripts_logs/" + exp_name
    os.makedirs(config_scripts_logs_path, exist_ok=True)
    with open(f"{config_scripts_logs_path}/{person_id}_pgd_loss_list.txt", "w") as f:
        # f.write(person_id + '\n')
        f.write(str(pgd_loss_list) + "\n")
        for index, loss_dict in enumerate(pgd_loss_list):
            f.write("index: " + str(index) + ", " + str(loss_dict) + "\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)
    