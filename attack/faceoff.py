import torch
import torch.nn.functional as F
import clip
from torchvision import transforms
import os
from PIL import Image
import json
from pathlib import Path
from diffusers import AutoencoderKL
import argparse
import json
import pdb
import cv2
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
from photomaker_clip import PhotoMakerIDEncoder
from face_diffuser_clip import FaceDiffuserCLIPImageEncoder
import random
import numpy as np
import time
import math
seed = 1
random.seed(seed) # python的随机种子一样
np.random.seed(seed) # numpy的随机种子一样
torch.manual_seed(seed) # 为cpu设置随机种子
torch.cuda.manual_seed_all(seed) # 为所有的gpu设置随机种子


def pgd_attack_refiner(model,
                model_type,
                perturbed_data,
                original_data,
                alpha,
                eps,
                attack_num,
                target_data,
                trans,
                min_eps,
                update_interval,
                noise_budget_refiner,
                refiner_type,
                loss_choice,
                w=0,
                weak_edge=200,
                strong_edge=300,
                mean_filter_size=3):
    if noise_budget_refiner == 1: # detect edge in original images
        ori_edges_list = []
        for ori_img in original_data:
            ori_img = ori_img.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            edges = cv2.Canny(ori_img, weak_edge, strong_edge)
            ori_edges_list.append(edges)
        # 如果refiner_type为pre，则将JND矩阵在ori_edges_list中的对应3*3均值滤波矩阵区域以外置为min_eps，在扰动过程中，JND保持不变
        if refiner_type == 'pre':
            print("refiner_type: pre")
            JND = np.full(perturbed_data.shape, min_eps) # dynamic noise budget matrix，非边缘区域，加弱噪声
            for idx, edges in enumerate(ori_edges_list): # 遍历perturbed_data中的每一个图片的边缘矩阵
                mean_filtered = cv2.blur(edges, (mean_filter_size, mean_filter_size)) # 3*3均值滤波矩阵获取边缘区域
                non_zero_indices = np.nonzero(mean_filtered) # non-zero positions，即边缘区域，加强噪声
                rows, cols = non_zero_indices[0], non_zero_indices[1]
                # 所有的边缘区域，都加强噪声，eps
                for i in range(len(rows)):
                    for j in range(0, 3):
                        JND[idx][j][rows[i], cols[i]] = eps
        # 如果refiner_type为mid，则将JND矩阵初始化为eps，在扰动过程中不断更新JND矩阵
        elif refiner_type == "mid0" or refiner_type == "mid1" or refiner_type == "mid2":
            print("refiner_type: mid")
            JND = np.full(perturbed_data.shape, eps) # dynamic noise budget matrix
        elif refiner_type == "post":
            print("refiner_type: post")
            JND = np.full(perturbed_data.shape, eps)
        else:
            raise ValueError('refiner_type choice must be one of pre, mid, and post')
    with torch.no_grad():
        if w == 0: # loss_type == 'x'
            tran_target = trans(target_data)
            if model_type == 'vae':
                target_image_embeds = model.encode(tran_target).latent_dist.sample() * model.config.scaling_factor
            elif model_type == 'clip':
                target_image_embeds = model.encode_image(tran_target)
            elif model_type == 'photomaker_clip':
                target_image_embeds = model(tran_target)
            elif model_type == 'ipadapter':
                target_image_embeds = model(tran_target, output_hidden_states=True).hidden_states[-2]
            elif model_type == 'face_diffuser':
                target_image_embeds = model(tran_target.unsqueeze(0))
            else:
                raise ValueError('model type choice must be one of vae, clip, and photomaker_clip')
        elif w > 0 and w < 1:
            original_data.requires_grad_(False)
            tran_original_data = trans(original_data)
            d_type = perturbed_data.dtype
            perturbed_data = (perturbed_data + (torch.rand(*perturbed_data.shape)*2*eps-eps).to(perturbed_data.device)).to(d_type)
            tran_target = trans(target_data)
            if model_type == 'vae':
                target_image_embeds = model.encode(tran_target).latent_dist.sample() * model.config.scaling_factor
                ori_embeds = model.encode(tran_original_data).latent_dist.sample() * model.config.scaling_factor
            elif model_type == 'photomaker_clip':
                target_image_embeds = model(tran_target)
                ori_embeds = model(tran_original_data)
            elif model_type == 'clip':
                target_image_embeds = model.encode_image(tran_target)
                ori_embeds = model.encode_image(tran_original_data)
            elif model_type == 'ipadapter':
                target_image_embeds = model(tran_target, output_hidden_states=True).hidden_states[-2]
                ori_embeds = model(tran_original_data, output_hidden_states=True).hidden_states[-2]
            elif model_type == 'face_diffuser':
                target_image_embeds = model(tran_target.unsqueeze(0))
                ori_embeds = model(tran_original_data.unsqueeze(0))
            else:
                raise ValueError('model type choice must be one of vae, clip, and photomaker_clip')
        elif w == 1: # loss_type == 'd'
            original_data.requires_grad_(False)
            tran_original_data = trans(original_data)
            d_type = perturbed_data.dtype
            perturbed_data = (perturbed_data + (torch.rand(*perturbed_data.shape)*2*eps-eps).to(perturbed_data.device)).to(d_type)
            if model_type == 'vae':
                ori_embeds = model.encode(tran_original_data).latent_dist.sample() * model.config.scaling_factor
            elif model_type == 'photomaker_clip':
                ori_embeds = model(tran_original_data)
            elif model_type == 'clip':
                ori_embeds = model.encode_image(tran_original_data)
            elif model_type == 'ipadapter':
                ori_embeds = model(tran_original_data, output_hidden_states=True).hidden_states[-2]
            elif model_type == 'face_diffuser':
                ori_embeds = model(tran_original_data.unsqueeze(0))
            else:
                raise ValueError('model type choice must be one of vae, clip, and photomaker_clip')
        else:
            raise ValueError('w must be in [0, 1]')
    Loss_dict = {}
    for k in range(0, attack_num):
        perturbed_data.requires_grad_()
        tran_perturbed_data = trans(perturbed_data)
        if w == 0: # loss_type == 'x'
            if model_type == 'vae':
                adv_image_embeds = model.encode(tran_perturbed_data).latent_dist.sample() * model.config.scaling_factor
            elif model_type == 'clip':
                adv_image_embeds = model.encode_image(tran_perturbed_data)
            elif model_type == 'photomaker_clip':
                adv_image_embeds = model(tran_perturbed_data)
            elif model_type == 'ipadapter':
                adv_image_embeds = model(tran_perturbed_data, output_hidden_states=True).hidden_states[-2]
            elif model_type == 'face_diffuser':
                adv_image_embeds = model(tran_perturbed_data.unsqueeze(0))
            else:
                raise ValueError('model type choice must be one of vae, clip, ipadapter, and photomaker_clip')
            if loss_choice == 'mse':
                Loss = F.mse_loss(adv_image_embeds, target_image_embeds, reduction="mean")
            elif loss_choice == 'cosine':
                Loss = -F.cosine_similarity(adv_image_embeds, target_image_embeds, -1).mean()
            else:
                raise ValueError('Loss choice must be one of mse or cosine')
        elif w == 1: # loss_type == 'd'
            if model_type == 'vae':
                adv_image_embeds = model.encode(tran_perturbed_data).latent_dist.sample() * model.config.scaling_factor
            elif model_type == 'photomaker_clip':
                adv_image_embeds = model(tran_perturbed_data)
            elif model_type == 'clip':
                adv_image_embeds = model.encode_image(tran_perturbed_data)
            elif model_type == 'ipadapter':
                adv_image_embeds = model(tran_perturbed_data, output_hidden_states=True).hidden_states[-2]
            elif model_type == 'face_diffuser':
                adv_image_embeds = model(tran_perturbed_data.unsqueeze(0))
            else:
                raise ValueError('model type choice must be one of vae, clip, ipadapter, and photomaker_clip')
            if loss_choice == 'mse':
                Loss = -F.mse_loss(adv_image_embeds, ori_embeds, reduction="mean")
            elif loss_choice == 'cosine':
                Loss = F.cosine_similarity(adv_image_embeds, ori_embeds, -1).mean()
            else:
                raise ValueError('Loss choice must be one of mse or cosine')
        elif w > 0 and w < 1:
            if model_type == 'vae':
                adv_image_embeds = model.encode(tran_perturbed_data).latent_dist.sample() * model.config.scaling_factor
            elif model_type == 'photomaker_clip':
                adv_image_embeds = model(tran_perturbed_data)
            elif model_type == 'clip':
                adv_image_embeds = model.encode_image(tran_perturbed_data)
            elif model_type == 'ipadapter':
                adv_image_embeds = model(tran_perturbed_data, output_hidden_states=True).hidden_states[-2]
            elif model_type == 'face_diffuser':
                adv_image_embeds = model(tran_perturbed_data.unsqueeze(0))
            else:
                raise ValueError('model type choice must be one of vae, clip, ipadapter, and photomaker_clip')
            if loss_choice == 'mse':
                Loss_x = F.mse_loss(adv_image_embeds, target_image_embeds, reduction="mean")
                Loss_d = -F.mse_loss(adv_image_embeds, ori_embeds, reduction="mean")
            elif loss_choice == 'cosine':
                Loss_x = -F.cosine_similarity(adv_image_embeds, target_image_embeds, -1).mean()
                Loss_d = F.cosine_similarity(adv_image_embeds, ori_embeds, -1).mean()
            else:
                raise ValueError('Loss choice must be one of mse or cosine')
            Loss = (1 - w) * Loss_x + w * Loss_d # [0, 4], 1 + 1 + 1 * (1 + 1) = 4 => 1 - 1 + 1 * (1 - 1) = 0
        else:
            raise ValueError('w must be in [0, 1]')
        # save loss
        if w > 0 and w < 1:
            Loss_dict[k] = [Loss.item(), Loss_x.item(), Loss_d.item()]
        else:
            Loss_dict[k] = Loss.item()
        if k % 10 == 0:
            print("k:{} th, Loss:{}".format(k, Loss))
            # 如果refiner的类型是mid0，那么就根据扰动过程中的边缘区域更新JND，直接降低到min_eps
            if noise_budget_refiner == 1 and (refiner_type == "mid0" or refiner_type == "mid2"):
                if k != 0 and k % update_interval == 0:
                    for idx, adv_img in enumerate(perturbed_data):
                        adv_img = adv_img.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                        edges = cv2.Canny(adv_img, weak_edge, strong_edge) # edges in adversarial images
                        edges = edges - ori_edges_list[idx]
                        mean_filtered = cv2.blur(edges, (mean_filter_size, mean_filter_size))
                        non_zero_indices = np.nonzero(mean_filtered) # non-zero positions
                        rows, cols = non_zero_indices[0], non_zero_indices[1]
                        for i in range(len(rows)):
                            for j in range(0, 3):
                                JND[idx][j][rows[i], cols[i]] = min_eps
                        print("idx:{}, min JND:{}, mean JND:{}".format(idx, np.min(JND[idx]), np.mean(JND[idx])))
            # 如果refiner的类型是mid1，那么就根据扰动过程中的边缘区域更新JND，逐渐降低
            elif noise_budget_refiner == 1 and refiner_type == "mid1": 
                drop_value = math.floor((eps - min_eps) / (attack_num / update_interval - 2)) # 1 = (16 -8) / (100 / 10 - 2); 1 = (16 - 12) / (50 / 10 - 2); 2 = (16 - 8) / (50 / 10 -2)
                if k != 0 and k % update_interval == 0:
                    for idx, adv_img in enumerate(perturbed_data):
                        adv_img = adv_img.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                        edges = cv2.Canny(adv_img, weak_edge, strong_edge) # edges in adversarial images
                        edges = edges - ori_edges_list[idx]
                        mean_filtered = cv2.blur(edges, (mean_filter_size, mean_filter_size))
                        non_zero_indices = np.nonzero(mean_filtered) # non-zero positions
                        rows, cols = non_zero_indices[0], non_zero_indices[1]
                        for i in range(len(rows)):
                            for j in range(0, 3):
                                JND[idx][j][rows[i], cols[i]] = max(JND[idx][j][rows[i], cols[i]] - drop_value, min_eps)
                        print("idx:{}, min JND:{}, mean JND:{}".format(idx, np.min(JND[idx]), np.mean(JND[idx])))
        grad = torch.autograd.grad(Loss, perturbed_data)[0]
        adv_perturbed_data = perturbed_data - alpha * grad.sign()
        # 只有"mid" or "pre"两种类型才会在扰动过程中使用JND矩阵
        if noise_budget_refiner == 1 and refiner_type != "post":
            et = (adv_perturbed_data - original_data).clamp(-255, 255).to(torch.float32).cpu().detach().numpy()
            et = np.minimum(np.maximum(et, -JND), JND)
            et = torch.from_numpy(et).to(model.device).to(model.dtype)
        else:
            et = torch.clamp(adv_perturbed_data - original_data, min=-eps, max=+eps)
        perturbed_data = torch.clamp(original_data + et, min=torch.min(original_data), max=torch.max(original_data)).detach().clone()
    # 如果refiner_type为post，那么，需要对扰动后的图像的进行边缘检测并抑制新出现的边缘区域
    if noise_budget_refiner == 1 and (refiner_type == "post" or refiner_type == "mid2"):
        # 获取新增边缘的JND矩阵
        for idx, adv_img in enumerate(perturbed_data):
            adv_img = adv_img.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            edges = cv2.Canny(adv_img, weak_edge, strong_edge) # edges in adversarial images
            edges = edges - ori_edges_list[idx]
            mean_filtered = cv2.blur(edges, (mean_filter_size, mean_filter_size))
            non_zero_indices = np.nonzero(mean_filtered) # non-zero positions
            rows, cols = non_zero_indices[0], non_zero_indices[1]
            for i in range(len(rows)):
                for j in range(0, 3):
                    JND[idx][j][rows[i], cols[i]] = min_eps # 新增边缘区域抑制
            print("idx:{}, min JND:{}, mean JND:{}".format(idx, np.min(JND[idx]), np.mean(JND[idx])))
        # 使用JND矩阵来更新et（第一行代码注释了因为最后一次循环已经获取了et）
        # et = (adv_perturbed_data - original_data).clamp(-255, 255).to(torch.float32).cpu().detach().numpy()
        et = et.clamp(-255, 255).to(torch.float32).cpu().detach().numpy()
        et = np.minimum(np.maximum(et, -JND), JND)
        et = torch.from_numpy(et).to(model.device).to(model.dtype)
        # 使用et更新对抗样本
        perturbed_data = torch.clamp(original_data + et, min=torch.min(original_data), max=torch.max(original_data)).detach().clone()
    return perturbed_data.cpu(), Loss_dict

def load_data(data_dir, image_size=512, resample=2):
    import numpy as np
    def image_to_numpy(image):
        return np.array(image).astype(np.uint8)
    # more robust loading to avoid loaing non-image files
    images = [] 
    for i in list(Path(data_dir).iterdir()):
        if not i.suffix in [".jpg", ".png", ".jpeg"]:
            continue
        else:
            images.append(image_to_numpy(Image.open(i).convert("RGB")))
    images = [Image.fromarray(i).resize((image_size, image_size), resample) for i in images]
    images = np.stack(images)
    # from B x H x W x C to B x C x H x W
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    assert images.shape[-1] == images.shape[-2]
    return images

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

def main(args):
    print(args)
    if args.prior_generation_precision == "fp32":
        torch_dtype = torch.float32
    elif args.prior_generation_precision == "fp16":
        torch_dtype = torch.float16
    elif args.prior_generation_precision == "bf16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError("prior_generation_precision must be one of [fp32, fp16, bf16]")
        
    if args.model_type == 'clip':
        model, _ = clip.load(args.pretrained_model_name_or_path, device=args.device)
        model.to(torch_dtype)
    elif args.model_type == 'vae':
        model = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision='bf16', torch_dtype=torch_dtype).to(args.device)
    elif args.model_type == 'photomaker_clip':
        model = PhotoMakerIDEncoder()
        state_dict = torch.load(args.pretrained_model_name_or_path, map_location="cpu")
        model.load_state_dict(state_dict['id_encoder'], strict=True)
        model.to(args.device, dtype=torch_dtype)
    elif args.model_type == 'ipadapter':
        model = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path).to(args.device, dtype=torch_dtype)
    elif args.model_type == "face_diffuser":
        model = FaceDiffuserCLIPImageEncoder.from_pretrained(args.pretrained_model_name_or_path,).to(args.device, dtype=torch_dtype)
    else:
        raise ValueError("model_type out of range")
    
    if args.resample_interpolation == 'BILINEAR':
        resample_interpolation = transforms.InterpolationMode.BILINEAR
    else:
        resample_interpolation = transforms.InterpolationMode.BICUBIC
        
    train_aug = [
        transforms.Resize(args.model_input_size, interpolation=resample_interpolation),
        transforms.CenterCrop(args.model_input_size) if args.center_crop else transforms.RandomCrop(args.model_input_size),
    ]
    tensorize_and_normalize = [
        transforms.Normalize([0.5*255]*3,[0.5*255]*3),
    ]
    all_trans = train_aug + tensorize_and_normalize
    all_trans = transforms.Compose(all_trans)
    print("all_trans:{}".format(all_trans))
    
    if args.noise_budget_refiner == 1:
        adv_image_dir_name = args.model_type + '_' + os.path.split(args.data_dir)[-1] + '_' + args.loss_choice + '_w' + str(args.w) + '_num' \
            + str(args.attack_num) + '_alpha' + str(args.alpha) + '_eps' + str(args.eps) + '_input' + str(args.input_size) \
            + '_' + str(args.model_input_size) + '_' + args.target_type + '_refiner' + str(args.noise_budget_refiner) + '_' + args.refiner_type \
            +  '_edge' + str(args.strong_edge) + '-' + str(args.weak_edge) + '_filter' + str(args.mean_filter_size) \
            + '_min-eps'+ str(args.min_eps) + '_interval' + str(args.update_interval)
        save_folder = os.path.join(args.save_dir, adv_image_dir_name)
    else:
        adv_image_dir_name = args.model_type + '_' + os.path.split(args.data_dir)[-1] + '_' + args.loss_choice + '_w' + str(args.w) + '_num' \
            + str(args.attack_num) + '_alpha' + str(args.alpha) + '_eps' + str(args.eps) + '_input' + str(args.input_size) \
            + '_' + str(args.model_input_size) + '_' + args.target_type + '_refiner' + str(args.noise_budget_refiner)
        save_folder = os.path.join(args.save_dir, adv_image_dir_name)
    resampling = {'NEAREST': 0, 'BILINEAR': 2, 'BICUBIC': 3}
    for person_id in sorted(os.listdir(args.data_dir)):
        person_folder = os.path.join(args.data_dir, person_id, args.input_name)
        clean_data = load_data(person_folder, args.input_size, resampling[args.resample_interpolation])
        
        if args.target_type == 'max':
            with open(args.max_distance_json, "r", encoding="utf-8") as f:
                max_dist_dict = json.load(f)
            target_folder_name = max_dist_dict[person_id]
            targeted_image_folder = os.path.join(args.data_dir_for_target_max, target_folder_name, args.input_name)
        elif args.target_type == 'yingbu':
            targeted_image_folder = './target_images/yingbu'
        elif args.target_type == 'mist':
            targeted_image_folder = './target_images/mist'
        elif args.target_type == 'colored_mist':
            targeted_image_folder = './target_images/colored_mist'
        elif args.target_type == 'gray':
            targeted_image_folder = './target_images/gray'
        else:
            raise ValueError("target_type out of range")
        target_data = load_data(targeted_image_folder, args.input_size, resampling[args.resample_interpolation]).to(dtype=torch_dtype)
        
        original_data = clean_data.detach().clone().to(args.device).requires_grad_(False).to(dtype=torch_dtype)
        perturbed_data = clean_data.to(dtype=torch_dtype).to(args.device).requires_grad_(True)
        target_data = target_data.to(args.device).requires_grad_(False)
        
        adv_data, Loss_dict = pgd_attack_refiner(model,
                            args.model_type,
                            perturbed_data,
                            original_data,
                            args.alpha,
                            args.eps,
                            args.attack_num,
                            target_data,
                            all_trans,
                            args.min_eps,
                            args.update_interval,
                            args.noise_budget_refiner,
                            args.refiner_type,
                            args.loss_choice,
                            args.w,
                            args.weak_edge,
                            args.strong_edge,
                            args.mean_filter_size)
        # save image
        savepath = os.path.join(save_folder, person_id)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        save_image(savepath, person_folder, adv_data)
        loss_save_path = f"./outputs/config_scripts_logs/{adv_image_dir_name}"
        os.makedirs(loss_save_path, exist_ok=True)
        with open(os.path.join(loss_save_path, "all_loss.txt"), mode="a", encoding="utf-8") as f:
            f.write("Person_id: " + str(person_id) + '\n')
            for key in Loss_dict.keys():
                f.write("Epoch: " + str(key) + ", Loss: " + str(Loss_dict[key]) + '\n')    
    return

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="adversarial attacks for customization models")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        required=True,
        help="select a cuda",
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default="bf16",
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--attack_num",
        type=int,
        default=200,
        required=True,
        help = "attack number"
    )
    parser.add_argument(
        "--loss_choice",
        type=str,
        default="cosine",
        required=True,
        help = "cosine or mse"
    )
    parser.add_argument(
        "--w",
        type=float,
        default=0.5,
        required=True,
        help = "w is used to adapt Targeted Loss and Deviation Loss"
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=6,
        required=True,
        help = "step size"
    )
    parser.add_argument(
        "--eps",
        type=int,
        default=16,
        required=True,
        help = "noise budget"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        required=True,
        help = "adversarial image size"
    )
    parser.add_argument(
        "--model_input_size",
        type=int,
        default=224,
        required=True,
        help = "model input image size"
    )
    parser.add_argument(
        "--center_crop",
        type=int,
        default=1,
        required=False,
        help = "center crop or not"
    )
    parser.add_argument(
        "--resample_interpolation",
        type=str,
        default='BILINEAR',
        required=True,
        help = "resample interpolation of resize, clip is BICUBIC"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets/mini-VGGFace2",
        required=True,
        help = "path to clean images"
    )
    parser.add_argument(
        "--input_name",
        type=str,
        default="set_B",
        required=True,
        help = "subfolder under data dir"
    )
    parser.add_argument(
        "--data_dir_for_target_max",
        type=str,
        default="./datasets/VGGFace2",
        required=False,
        help = "path to all clean images"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/photomaker/adversarial_images",
        required=True,
        help = "save path"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default='photomaker_clip',
        required=True,
        help = "vae, clip, ipadapter, photomaker_clip"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="./pretrains/photomaker-v1.bin",
        required=True,
        help = "ViT-L/14, ./pretrains/photomaker-v1.bin, /home/humw/Pretrain/stable-diffusion-2-1-base"
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default='max',
        required=True,
        help = "target image choice, max means photomaker clip distance, yingbu means target opera make-up yingbu"
    )
    parser.add_argument(
        "--max_distance_json",
        type=str,
        default="./customization/target_model/PhotoMaker/VGGFace2_max_photomaker_clip_distance.json",
        required=True,
        help = "./customization/target_model/PhotoMaker/VGGFace2_max_photomaker_clip_distance.json"
    )
    parser.add_argument(
        "--mean_filter_size",
        type=int,
        default=3,
        required=True,
        help = "blur size in refiner"
    )
    parser.add_argument(
        "--strong_edge",
        type=int,
        default=300,
        required=True,
        help = "strong edge threshold of edge detector"
    )
    parser.add_argument(
        "--weak_edge",
        type=int,
        default=200,
        required=True,
        help = "weak edge threshold of edge detector"
    )
    parser.add_argument(
        "--min_eps",
        type=int,
        default=12,
        required=True,
        help = "min_eps of refiner"
    )
    parser.add_argument(
        "--update_interval",
        type=int,
        default=40,
        required=True,
        help = "update_interval"
    )
    parser.add_argument(
        "--noise_budget_refiner",
        type=int,
        default=1,
        required=True,
        help = "with noise budget refiner"
    )
    parser.add_argument(
        "--refiner_type",
        type=str,
        default="mid",
        required=True,
        help = "pre, mid0, mid1, mid2, post"
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # t1 = time.time()
    main(args)
    # t2 = time.time()
    # print('TIME COST: %.6f'%(t2-t1))
    # with open(file="time_costs.txt", mode='a') as f:
    #     f.write(str(t2-t1) + '\n')
