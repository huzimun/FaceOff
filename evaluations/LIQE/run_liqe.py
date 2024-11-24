import pyiqa
import torch
import torch
import clip
from PIL import Image
import os
import os
import torch
import argparse
import pdb
import json
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='IMS_clip evaluation')
    parser.add_argument('--data_dirs', type=str, default='', required=True, help='path of folders to perturbed image')
    parser.add_argument('--prompts', type=str, required=True, help='prompts string')
    parser.add_argument('--save_dir', type=str, required=True, help='dir to save evaluation results')
    parser.add_argument('--scene', type=str, default='perturbed_output', required=False, help='choose perturbed_output or original_output')
    parser.add_argument('--device', type=str, default='cuda', required=True, help='cuda or cpu')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    model = pyiqa.create_metric('liqe', as_loss=False, device=device) #Re-trained on the official set of KonIQ-10k
    prompts = args.prompts.split(';')
    prompt_paths_list = []
    for i in range(0, len(prompts)):
        prompt_paths_list.append(prompts[i].replace(' ', '_'))
    persons_list = os.listdir(args.data_dirs)
    # 遍历对抗图片列表
    liqe_list = [0.0 for _ in range(0, len(persons_list))]
    with torch.no_grad():
        for k, person_id in enumerate(persons_list):
            tmp_liqe_list = list()

            # 遍历提示词列表
            for prompt_name in prompt_paths_list:
                image_path = os.path.join(args.data_dirs, person_id, prompt_name)
                # 获取id image的平均特征
                ave_liqe = 0
                img_list = os.listdir(image_path)
                for img in img_list:
                    img_path = os.path.join(image_path, img)
                    score = model(target=img_path)
                    ave_liqe += score
                ave_liqe /= len(img_list)
                if ave_liqe == None:
                    tmp_liqe_list.append(0)
                else:
                    tmp_liqe_list.append(ave_liqe.item())
            print("tmp_ism_list:{}".format(str(tmp_liqe_list)))
            liqe_list[k] = sum(tmp_liqe_list) * 1.0 / len(tmp_liqe_list)
    print("ism_list:{}".format(liqe_list))
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.scene + '_'  + 'LIQE' + '.txt')
    with open(save_path, 'w') as f:
        f.write(str(liqe_list) + '\n')
        f.write(str(sum(liqe_list) * 1.0 / len(liqe_list)) + '\n')
    return
    
            
if __name__ == '__main__':
    main()