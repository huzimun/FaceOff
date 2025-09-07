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
from photomaker_clip import PhotoMakerIDEncoder1
from face_diffuser_clip import FaceDiffuserCLIPImageEncoder
import random
import numpy as np
import time
import math
from ip_adapter.resampler import Resampler
from ip_adapter.ip_adapter import ImageProjModel

seed = 1
random.seed(seed) # python的随机种子一样
np.random.seed(seed) # numpy的随机种子一样
torch.manual_seed(seed) # 为cpu设置随机种子
torch.cuda.manual_seed_all(seed) # 为所有的gpu设置随机种子

# 多模型攻击，对多个图像编码器进行攻击
def pgd_ensemble_attack(model_dict, # 模型池, key为模型类型，value为模型
                perturbed_data,
                origin_data,
                alpha,
                eps,
                attack_num,
                target_data,
                eot_trans_list,
                loss_choice,
                w,
                image_proj_model_dict):
    with torch.no_grad():
        origin_data.requires_grad_(False)
        eot_trans_target_data_list = list()
        eot_trans_origin_data_list = list()
        for trans in eot_trans_list:
            tran_target_data = trans(target_data)
            tran_origin_data = trans(origin_data)
            eot_trans_target_data_list.append(tran_target_data)
            eot_trans_origin_data_list.append(tran_origin_data)
        d_type = perturbed_data.dtype
        perturbed_data = (perturbed_data + (torch.rand(*perturbed_data.shape)*2*eps-eps).to(perturbed_data.device)).to(d_type)
        target_embeds_dict = {} # 目标图像的编码列表
        origin_embeds_dict = {} # 原始图像的编码列表
        for k in model_dict.keys():
            model_type = k
            model = model_dict[k]
            eot_trans_target_embeds_list = list()
            eot_trans_origin_embeds_list = list()
            for i in range(0, len(eot_trans_list)):
                tran_target_data = eot_trans_target_data_list[i]
                tran_origin_data = eot_trans_origin_data_list[i]
                if model_type == 'clip' or model_type == 'ViT-B32' or model_type == 'ViT-B16' or model_type == 'ViT-L14':
                    target_embeds = model.encode_image(tran_target_data)
                    origin_embeds = model.encode_image(tran_origin_data)
                elif model_type == 'photomaker':
                    target_embeds = model(tran_target_data)
                    origin_embeds = model(tran_origin_data)
                elif model_type == 'ipadapter-plus':
                    target_embeds = model(tran_target_data, output_hidden_states=True).hidden_states[-2]
                    origin_embeds = model(tran_origin_data, output_hidden_states=True).hidden_states[-2]
                    if image_proj_model_dict[model_type] is not None:
                        proj_origin_embeds = image_proj_model_dict[model_type](origin_embeds)
                        origin_embeds = proj_origin_embeds
                        proj_target_embeds = image_proj_model_dict[model_type](target_embeds)
                        target_embeds = proj_target_embeds
                elif model_type == 'ipadapter':
                    target_embeds = model(tran_target_data, output_hidden_states=True).pooler_output
                    origin_embeds = model(tran_origin_data, output_hidden_states=True).pooler_output
                    if image_proj_model_dict[model_type] is not None:
                        proj_origin_embeds = image_proj_model_dict[model_type](origin_embeds)
                        origin_embeds = proj_origin_embeds
                        proj_target_embeds = image_proj_model_dict[model_type](target_embeds)
                        target_embeds = proj_target_embeds
                elif model_type == 'face_diffuser':
                    target_embeds = model(tran_target_data.unsqueeze(0))
                    origin_embeds = model(tran_origin_data.unsqueeze(0))
                else:
                    raise ValueError('model type choice must be one of clip, and photomaker')
                eot_trans_target_embeds_list.append(target_embeds)
                eot_trans_origin_embeds_list.append(origin_embeds)
            target_embeds_dict[k] = eot_trans_target_embeds_list
            origin_embeds_dict[k] = eot_trans_origin_embeds_list
    # pdb.set_trace()
    Loss_dict = {}
    for epoch in range(0, attack_num):
        perturbed_data.requires_grad_()
        eot_trans_perturbed_data_list = list()
        for trans in eot_trans_list:
            tran_perturbed_data = trans(perturbed_data)
            eot_trans_perturbed_data_list.append(tran_perturbed_data)
        Loss_x_ = []
        Loss_d_ = []
        for k in model_dict.keys():
            model_type = k
            model = model_dict[k]
            loss_x_list = list()
            loss_d_list = list()
            for i in range(0, len(eot_trans_list)): # 每个trans下的loss累加作为该模型对应的loss
                tran_perturbed_data = eot_trans_perturbed_data_list[i]
                wi = 1
                if model_type == 'photomaker':
                    adv_embeds = model(tran_perturbed_data)
                elif model_type == 'clip' or model_type == 'ViT-B32' or model_type == 'ViT-B16' or model_type == 'ViT-L14':
                    adv_embeds = model.encode_image(tran_perturbed_data)
                elif model_type == 'ipadapter-plus':
                    adv_embeds = model(tran_perturbed_data, output_hidden_states=True).hidden_states[-2]
                    if image_proj_model_dict[model_type] is not None:
                        proj_origin_embeds = image_proj_model_dict[model_type](adv_embeds)
                        adv_embeds = proj_origin_embeds
                elif model_type == 'ipadapter':
                    adv_embeds = model(tran_perturbed_data, output_hidden_states=True).pooler_output
                    if image_proj_model_dict[model_type] is not None:
                        proj_origin_embeds = image_proj_model_dict[model_type](adv_embeds)
                        adv_embeds = proj_origin_embeds
                elif model_type == 'face_diffuser':
                    adv_embeds = model(tran_perturbed_data.unsqueeze(0))
                else:
                    raise ValueError('model type choice must be one of clip, ipadapter, and photomaker')
                
                target_embeds_list = target_embeds_dict[k]
                origin_embeds_list = origin_embeds_dict[k]
                target_embeds = target_embeds_list[i]
                origin_embeds = origin_embeds_list[i]
                if loss_choice == 'mse':
                    Loss_x = wi * F.mse_loss(adv_embeds, target_embeds, reduction="mean")
                    Loss_d = -F.mse_loss(adv_embeds, origin_embeds, reduction="mean")
                elif loss_choice == 'cosine':
                    Loss_x = -F.cosine_similarity(adv_embeds, target_embeds, -1).mean()
                    Loss_d = F.cosine_similarity(adv_embeds, origin_embeds, -1).mean()
                else:
                    raise ValueError('Loss choice must be one of mse or cosine')
                loss_x_list.append(Loss_x)
                loss_d_list.append(Loss_d)
            # EOT，对每个trans下的loss取平均
            mean_Loss_x = torch.stack(loss_x_list).mean()
            mean_Loss_d = torch.stack(loss_d_list).mean()
            Loss_x_.append(mean_Loss_x)
            Loss_d_.append(mean_Loss_d)
        # pdb.set_trace()
        Loss_x_ = torch.stack(Loss_x_).view(1, len(model_dict.keys()))
        Loss_d_ = torch.stack(Loss_d_).view(1, len(model_dict.keys()))
        Loss_ = (1 - w) * Loss_x_ + w * Loss_d_ # [0, 4], 1 + 1 + 1 * (1 + 1) = 4 => 1 - 1 + 1 * (1 - 1) = 0
        Loss_ = Loss_.mean()
        # save loss
        Loss_dict[epoch] = [Loss_.item(), Loss_x_.mean().item(), Loss_d_.mean().item()]
        if epoch % 10 == 0:
            print("epoch:{} th, Loss:{}".format(epoch, Loss_.item()))
        grad = torch.autograd.grad(Loss_, perturbed_data)[0]
        adv_perturbed_data = perturbed_data - alpha * grad.sign()
        et = torch.clamp(adv_perturbed_data - origin_data, min=-eps, max=+eps)
        perturbed_data = torch.clamp(origin_data + et, min=torch.min(origin_data), max=torch.max(origin_data)).detach().clone()
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
    model_type_list = args.model_type.split(',')
    model_path_list = args.pretrained_model_name_or_path.split(',')
    model_dict = {} # key为模型类型，value为模型
    image_proj_model_dict = {} # key为ipadapter模型类型，value为投影层模型
    for idx in range(0, len(model_type_list)):
        print("model_type:{}, pretrained_model_name_or_path:{}".format(model_type_list[idx], model_path_list[idx]))
        model_type = model_type_list[idx]
        model_path = model_path_list[idx]
        if model_type == 'clip' or model_type == 'ViT-B32' or model_type == 'ViT-B16' or model_type == 'ViT-L14':
            model, _ = clip.load(model_path, device=args.device)
            model.to(torch_dtype)
        if model_type == 'photomaker':
            if args.mode == "idprotector": # no visual projection
                model = PhotoMakerIDEncoder1()
                state_dict = torch.load(model_path, map_location="cpu")
                model.load_state_dict(state_dict['id_encoder'], strict=False)
                model.to(args.device, dtype=torch_dtype)
            else:
                model = PhotoMakerIDEncoder()
                state_dict = torch.load(model_path, map_location="cpu")
                model.load_state_dict(state_dict['id_encoder'], strict=False)
                model.to(args.device, dtype=torch_dtype)
        elif model_type == 'ipadapter' or model_type == 'ipadapter-plus':
            # pdb.set_trace()
            # model = CLIPVisionModelWithProjection.from_pretrained(model_path).to(args.device, dtype=torch_dtype)
            # ipadapter_path = "/data1/humw/Pretrains/IP-Adapter/models/image_encoder"
            # model = CLIPVisionModelWithProjection.from_pretrained(ipadapter_path).to(dtype=torch_dtype).eval().requires_grad_(False)
            if args.mode == "idprotector": # with visual projection
                # load SDXL pipeline
                from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline
                from ip_adapter import IPAdapterPlusXL
                from ip_adapter import IPAdapterXL
                
                base_model_path = "/home/humw/Pretrains/stabilityai/stable-diffusion-xl-base-1.0"
                pipe = StableDiffusionXLCustomPipeline.from_pretrained(
                    base_model_path,
                    # torch_dtype=torch.float16,
                    add_watermarker=False,
                )
                # 加载投影层参数
                if model_type == "ipadapter-plus":
                    ip_ckpt = "/home/humw/Pretrains/h94/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin"
                    ip_model = IPAdapterPlusXL(pipe, model_path, ip_ckpt, "cpu", num_tokens=16)
                    image_proj_model = ip_model.image_proj_model.to(args.device).to(dtype=torch_dtype)
                elif model_type == "ipadapter":
                    ip_ckpt = "/home/humw/Pretrains/h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
                    ip_model = IPAdapterXL(pipe, model_path, ip_ckpt, "cpu",)
                    image_proj_model = ip_model.image_encoder.visual_projection.to(args.device).to(dtype=torch_dtype)
                else:
                    raise ValueError("model_type out of range")
                model = ip_model.image_encoder.vision_model.to(args.device).to(dtype=torch_dtype)
                image_proj_model_dict[model_type] = image_proj_model
                del pipe
            else:
                image_proj_model = None
        elif model_type == "face_diffuser":
            model = FaceDiffuserCLIPImageEncoder.from_pretrained(model_path,).to(args.device, dtype=torch_dtype)
        else:
            raise ValueError("model_type out of range")
        model_dict[model_type] = model
    if args.resample_interpolation == 'BILINEAR':
        resample_interpolation = transforms.InterpolationMode.BILINEAR
    else:
        resample_interpolation = transforms.InterpolationMode.BICUBIC
    
    eot_trans_list = []
    for tmp in args.eot_trans_types.split(','):
        train_aug_for_clip = [
            transforms.Resize(224, interpolation=resample_interpolation),
            transforms.CenterCrop(224) if args.center_crop else transforms.RandomCrop(224),
        ]
        tensorize_and_normalize = [
            transforms.Normalize([0.5*255]*3,[0.5*255]*3),
        ]
        defense_transform = [
        ]
        if tmp == 'gau':
            gau_filter = transforms.GaussianBlur(kernel_size=args.gau_kernel_size,)
            defense_transform = [gau_filter]
        elif tmp == 'hflip':
            hflip = transforms.RandomHorizontalFlip(p=0.5)
            defense_transform = [hflip]
        elif tmp == 'none':
            continue
        elif tmp == 'gau-hflip':
            gau_filter = transforms.GaussianBlur(kernel_size=args.gau_kernel_size,)
            hflip = transforms.RandomHorizontalFlip(p=0.5)
            defense_transform = [gau_filter] + [hflip]
        else:
            raise ValueError("eot_trans_types out of range")
        trans_for_clip = train_aug_for_clip + defense_transform + tensorize_and_normalize
        trans_for_clip = transforms.Compose(trans_for_clip)
        print("all_trans:{}".format(trans_for_clip))
        eot_trans_list.append(trans_for_clip)
    print("eot_trans_list:{}", eot_trans_list)
    
    adv_image_dir_name = args.adversarial_folder_name
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
        
        origin_data = clean_data.detach().clone().to(args.device).requires_grad_(False).to(dtype=torch_dtype)
        perturbed_data = clean_data.to(dtype=torch_dtype).to(args.device).requires_grad_(True)
        target_data = target_data.to(args.device).requires_grad_(False)
        
        adv_data, Loss_dict = pgd_ensemble_attack(model_dict, # 模型池, key为模型类型，value为模型
                perturbed_data,
                origin_data,
                args.alpha*255,
                args.eps,
                args.attack_num,
                target_data,
                eot_trans_list,
                args.loss_choice,
                args.w,
                image_proj_model_dict=image_proj_model_dict)
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
        "--mode", 
        type=str, 
        default="idprotector", 
        help="idprotector use projection image embeds for ip-adapter"
    )
    parser.add_argument(
        "--adversarial_folder_name",
        type=str,
        required=True,
        default="adversarial_folder_name",
        help="adversarial_folder_name",
    )
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
        type=float,
        default=0.005,
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
        "--center_crop",
        type=int,
        default=1,
        required=False,
        help = "center crop or not"
    )
    parser.add_argument(
        "--eot_trans_types",
        type=str,
        default="gau,none",
        required=False,
        help = "gau,hflip,gau-hflip,none"
    )
    parser.add_argument(
        "--gau_kernel_size",
        type=int,
        default=7,
        required=False,
        help = "gaussian kernel size"
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
        default='ipadapter,photomaker,face_diffuser',
        required=True,
        help = "clip, ipadapter, photomaker, face_diffuser"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/data1/humw/Pretrains/stable-diffusion-v1-5,/data1/humw/Pretrains/IP-Adapter/models/image_encoder,/data1/humw/Pretrains/photomaker-v1.bin,/data1/humw/Pretrains/clip-vit-large-patch14",
        required=True,
        help = "pretrained model path"
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
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
    # args = parse_args()
    # t1 = time.time()
    # main(args)
    # t2 = time.time()
    # print('TIME COST: %.6f'%(t2-t1))
    # with open(file="time_costs.txt", mode='a') as f:
    #     f.write(str(t2-t1) + '\n')
