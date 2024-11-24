import lpips
import torch
from PIL import Image
import os
import argparse
import pdb
import json
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description='IMS_lpips evaluation')
    parser.add_argument('--data_dirs', type=str, default='', required=True, help='path of folders to perturbed image')
    parser.add_argument('--save_dir', type=str, required=True, help='dir to save evaluation results')
    parser.add_argument('--sub_folder', type=str, default='', required=True, help='sub folder')
    parser.add_argument('--resolution', type=int, default=224, required=True, help='224 or 512')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    resize_tran = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    # 遍历对抗图片列表
    persons_list = os.listdir(args.data_dirs)
    os.makedirs(args.save_dir, exist_ok=True)
    with torch.no_grad():
        for k, person_id in enumerate(persons_list):
            save_person_dir = os.path.join(args.save_dir, person_id)
            os.makedirs(save_person_dir, exist_ok=True)
            # 遍历提示词列表
            image_path = os.path.join(args.data_dirs, person_id + '/' + args.sub_folder)
            # 获取id image的平均特征
            img_list = sorted(os.listdir(image_path))
            img_names = [
                    str(instance_path).split("/")[-1]
                    for instance_path in img_list
            ]
            resize_images = []
            for idx, img in enumerate(img_list):
                img_path = os.path.join(image_path, img)
                image1 = Image.open(img_path)
                image1_tensor = torch.tensor(np.array(image1)).permute(2, 0, 1)
                image1_tensor = resize_tran(image1_tensor)
                resize_images.append(image1_tensor)
            # pdb.set_trace()
            for img_pixel, img_name in zip(resize_images, img_names):
                save_img_path = os.path.join(save_person_dir, img_name)
                Image.fromarray(
                    # (img_pixel * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                    img_pixel.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                ).save(save_img_path)

    return
            
if __name__ == '__main__':
    main()