import torch
import torch.nn.functional as F
import clip
import numpy as np
import torchvision
from torchvision import transforms
import random
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


def pgd_attack_refiner(model,
                model_type,
                perturbed_data,
                original_data,
                alpha,
                eps,
                attack_num,
                loss_type,
                target_data,
                trans,
                min_JND_eps_rate,
                update_interval,
                noise_budget_refiner):
    if noise_budget_refiner == 1: # detect edge in original images
        JND = np.full(perturbed_data.shape, eps) # dynamic noise budget matrix
        ori_edges_list = []
        for ori_img in original_data:
            ori_img = ori_img.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            edges = cv2.Canny(ori_img, 200, 300)
            ori_edges_list.append(edges)
    with torch.no_grad():
        if loss_type == 'x': 
            tran_target = trans(target_data)
            if model_type == 'vae':
                target_image_embeds = model.encode(tran_target).latent_dist.sample() * model.config.scaling_factor
            elif model_type == 'clip':
                target_image_embeds = model.encode_image(tran_target)
            elif model_type == 'photomaker_clip':
                target_image_embeds = model(tran_target)
            elif model_type == 'ipadapter':
                target_image_embeds = model(tran_target, output_hidden_states=True).hidden_states[-2]
        else: # loss type == 'n'
            original_data.requires_grad_(False)
            target_data.requires_grad_(False)
            tran_target_data = trans(target_data)
            tran_original_data = trans(original_data)
            if model_type == 'vae':
                target_embeds = model.encode(tran_target_data).latent_dist.sample() * model.config.scaling_factor
                ori_embeds = model.encode(tran_original_data).latent_dist.sample() * model.config.scaling_factor
            elif model_type == 'photomaker_clip':
                target_embeds = model(tran_target_data)
                ori_embeds = model(tran_original_data)
            elif model_type == 'clip':
                target_embeds = model.encode_image(tran_target_data)
                ori_embeds = model.encode_image(tran_original_data)
            elif model_type == 'ipadapter':
                target_embeds = model(tran_target_data, output_hidden_states=True).hidden_states[-2]
                ori_embeds = model(tran_original_data, output_hidden_states=True).hidden_states[-2]
            else:
                raise ValueError('model type choice must be one of vae, clip, and photomaker_clip')
            target_embeds_noise = target_embeds - ori_embeds # target noise embeddings
    for k in range(0, attack_num):
        perturbed_data.requires_grad_()
        tran_perturbed_data = trans(perturbed_data)
        if loss_type == 'x':
            if model_type == 'vae':
                adv_image_embeds = model.encode(tran_perturbed_data).latent_dist.sample() * model.config.scaling_factor
            elif model_type == 'clip':
                adv_image_embeds = model.encode_image(tran_perturbed_data)
            elif model_type == 'photomaker_clip':
                adv_image_embeds = model(tran_perturbed_data)
            elif model_type == 'ipadapter':
                adv_image_embeds = model(tran_perturbed_data, output_hidden_states=True).hidden_states[-2]
            else:
                raise ValueError('model type choice must be one of vae, clip, ipadapter, and photomaker_clip')
            Loss = 1-F.cosine_similarity(adv_image_embeds, target_image_embeds, -1).mean()
        else: # loss_type == 'n'
            if model_type == 'vae':
                tmp_embeds = model.encode(tran_perturbed_data).latent_dist.sample() * model.config.scaling_factor
            elif model_type == 'photomaker_clip':
                tmp_embeds = model(tran_perturbed_data)
            elif model_type == 'clip':
                tmp_embeds = model.encode_image(tran_perturbed_data).latent_dist.sample() * model.config.scaling_factor
            elif model_type == 'ipadapter':
                tmp_embeds = model(tran_perturbed_data, output_hidden_states=True).hidden_states[-2]
            else:
                raise ValueError('model type choice must be one of vae, clip, ipadapter, and photomaker_clip')
            adv_embeds_noise = tmp_embeds - ori_embeds # target semantic difference
            Loss = 1-F.cosine_similarity(adv_embeds_noise, target_embeds_noise, -1).mean()  
        if k % 10 == 0:
            print("k:{} th, Loss:{}".format(k, Loss))
            if noise_budget_refiner == 1:
                if k != 0 and k % update_interval == 0:
                    for idx, adv_img in enumerate(perturbed_data):
                        adv_img = adv_img.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                        edges = cv2.Canny(adv_img, 200, 300) # edges in adversarial images
                        edges = edges - ori_edges_list[idx]
                        mean_filtered = cv2.blur(edges, (3, 3))
                        non_zero_indices = np.nonzero(mean_filtered) # non-zero positions
                        rows, cols = non_zero_indices[0], non_zero_indices[1]
                        for i in range(len(rows)):
                            for j in range(0, 3):
                                JND[idx][j][rows[i], cols[i]] = max(JND[idx][j][rows[i], cols[i]] - 1, eps * min_JND_eps_rate)
                        print("idx:{}, min JND:{}, mean JND:{}".format(idx, np.min(JND[idx]), np.mean(JND[idx])))
        grad = torch.autograd.grad(Loss, perturbed_data)[0]
        adv_perturbed_data = perturbed_data - alpha * grad.sign()
        if noise_budget_refiner == 1:
            et = (adv_perturbed_data - original_data).clamp(0, 255).to(torch.float32).cpu().detach().numpy()
            et = np.minimum(np.maximum(et, -JND), JND)
            et = torch.from_numpy(et).to(model.device).to(model.dtype)
        else:
            et = torch.clamp(adv_perturbed_data - original_data, min=-eps, max=+eps)
        perturbed_data = torch.clamp(original_data + et, min=torch.min(original_data), max=torch.max(original_data)).detach().clone()
    return perturbed_data.cpu()

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
    
    if args['prior_generation_precision'] == "fp32":
        torch_dtype = torch.float32
    elif args['prior_generation_precision'] == "fp16":
        torch_dtype = torch.float16
    elif args['prior_generation_precision'] == "bf16":
        torch_dtype = torch.bfloat16
        
    if args['model_type'] == 'clip':
        model, preprocess = clip.load(args['pretrained_model_name_or_path'], device=args['device'])
        model.to(torch_dtype)
    elif args['model_type'] == 'vae':
        model = AutoencoderKL.from_pretrained(args['pretrained_model_name_or_path'], subfolder="vae", revision='bf16', torch_dtype=torch_dtype).to(args['device'])
    elif args['model_type'] == 'photomaker_clip':
        model = PhotoMakerIDEncoder()
        state_dict = torch.load(args['pretrained_model_name_or_path'], map_location="cpu")
        model.load_state_dict(state_dict["id_encoder"], strict=True)
        model.to(args['device'], dtype=torch_dtype)
    elif args['model_type'] == 'ipadapter':
        model = CLIPVisionModelWithProjection.from_pretrained(args['pretrained_model_name_or_path']).to(args['device'], dtype=torch_dtype) # /home/humw/Pretrain/h94/IP-Adapter/models/image_encoder

    if args['resample_interpolation'] == 'BILINEAR':
        resample_interpolation = transforms.InterpolationMode.BILINEAR
    else:
        resample_interpolation = transforms.InterpolationMode.BICUBIC
        
    train_aug = [
        transforms.Resize(args['output_size'], interpolation=resample_interpolation),
        transforms.CenterCrop(args['output_size']) if args['center_crop'] else transforms.RandomCrop(args['output_size']),
    ]
    tensorize_and_normalize = [
        transforms.Normalize([0.5*255]*3,[0.5*255]*3),
    ]
    all_trans = train_aug + tensorize_and_normalize
    all_trans = transforms.Compose(all_trans)
    print("all_trans:{}".format(all_trans))
    
    with open(args['max_distance_json'], "r", encoding="utf-8") as f:
        max_dist_dict = json.load(f)
        
    save_folder = os.path.join(args['save_dir'], args['model_type'] + '_' + args['loss_type'] + '_num' + str(args['attack_num']) + '_alpha' + str(args['alpha']) + '_eps' + str(args['eps']) + '_input' + str(args['input_size']) + '_output' + str(args['output_size']) + '_' + args['target_type'] + '_refiner' + str(args['noise_budget_refiner']) + '_min-eps'+ str(int(args['min_JND_eps_rate']*args['eps'])))
    resampling = {'NEAREST': 0, 'BILINEAR': 2, 'BICUBIC': 3}
    for person_id in sorted(os.listdir(args['data_dir'])):
        person_folder = os.path.join(args['data_dir'], person_id, args['input_name'])
        clean_data = load_data(person_folder, args['input_size'], resampling[args['resample_interpolation']])
        
        if args['target_type'] == 'max':
            target_folder_name = max_dist_dict[person_id]
            targeted_image_folder = os.path.join(args['data_dir_for_target_max'], target_folder_name, args['input_name'])
        elif args['target_type'] == 'yingbu':
            targeted_image_folder = '/home/humw/Codes/FaceOff/target_images/yingbu'
        target_data = load_data(targeted_image_folder, args['input_size'], resampling[args['resample_interpolation']]).to(dtype=torch_dtype)
        
        original_data = clean_data.detach().clone().to(args['device']).requires_grad_(False).to(dtype=torch_dtype)
        perturbed_data = clean_data.to(dtype=torch_dtype).to(args['device']).requires_grad_(True)
        target_data = target_data.to(args['device']).requires_grad_(False)
        
        adv_data = pgd_attack_refiner(model,
                            args['model_type'],
                            perturbed_data,
                            original_data,
                            args['alpha'],
                            args['eps'],
                            args['attack_num'],
                            args['loss_type'],
                            target_data,
                            all_trans,
                            args['min_JND_eps_rate'],
                            args['update_interval'],
                            args['noise_budget_refiner'])
        
        savepath = os.path.join(save_folder, person_id)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        save_image(savepath, person_folder, adv_data)
    return

if __name__ == "__main__":
    args_path = '/home/humw/Codes/AAAI/FaceOff/args.json'
    f = open(args_path, 'r')
    args = json.load(f)
    main(args)
