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
    parser.add_argument('--emb_dirs', type=str, required=True, help='path of folders to clean image')
    parser.add_argument('--save_dir', type=str, required=True, help='dir to save evaluation results')
    parser.add_argument('--sub_folder', type=str, default='', required=False, help='set_B or none')
    parser.add_argument('--scene', type=str, default='perturbed_input', required=False, help='choose perturbed_output or original_output')
    parser.add_argument('--scene2', type=str, default='original_input', required=False, help='the other folder name')
    parser.add_argument('--model_name_or_path', type=str, default="/home/humw/Codes/FaceOff/evaluations/clip/ViT-B-32.pt", required=True, help='ViT-B/32')
    parser.add_argument('--resolution', type=int, default=224, required=True, help='512 or 224')
    parser.add_argument('--device', type=str, default='cuda', required=True, help='cuda or cpu')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    device = args.device if torch.cuda.is_available() else "cpu"
    lpips_model = lpips.LPIPS(net=args.model_name_or_path).to(device)
    persons_list = os.listdir(args.data_dirs)
    resize_tran = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    # 遍历对抗图片列表
    lpips_list = [0.0 for _ in range(0, len(persons_list))]
    with torch.no_grad():
        for k, person_id in enumerate(persons_list):
            list_id_path = os.path.join(args.emb_dirs, person_id, 'set_B')
            # 遍历提示词列表
            image_path = os.path.join(args.data_dirs, person_id + '/' + args.sub_folder)
            # 获取id image的平均特征
            ave_score = 0 # 记录一个提示词下输出
            id_img_list = sorted(os.listdir(list_id_path)) # 一定要sort，防止因为不同的前后缀导致的差错
            img_list = sorted(os.listdir(image_path))
            for idx, id_img in enumerate(id_img_list):
                id_img_path = os.path.join(list_id_path, id_img)
                img_path = os.path.join(image_path, img_list[idx])
                
                image1 = Image.open(id_img_path)
                image1_tensor = torch.tensor(np.array(image1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                image1_tensor = image1_tensor.to(device)

                image2 = Image.open(img_path)
                image2_tensor = torch.tensor(np.array(image2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                image2_tensor = image2_tensor.to(device)
                # w = min(image1_tensor.shape[-2], image2_tensor.shape[-2])
                # h = min(image1_tensor.shape[-1], image2_tensor.shape[-1])
                # size = min(w, h)
                image1_tensor = resize_tran(image1_tensor)
                image2_tensor = resize_tran(image2_tensor)
                distance = lpips_model(image1_tensor, image2_tensor).item()
                ave_score += distance
            lpips_list[k] = ave_score / len(id_img_list)
    print("lpips_list:{}".format(lpips_list))
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.scene + '_' + args.scene2 + '_' + 'LPIPS' + '.txt')
    with open(save_path, 'w') as f:
        f.write(str(lpips_list) + '\n')
        f.write(str(sum(lpips_list) * 1.0 / len(lpips_list)) + '\n')
    return
            
if __name__ == '__main__':
    main()