import torch
import clip
from PIL import Image
import os
import torch
import argparse
import pdb
import json
import torch.nn.functional as F
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='IMS_clip evaluation')
    parser.add_argument('--data_dirs', type=str, default='', required=True, help='path of folders to perturbed image')
    parser.add_argument('--emb_dirs', type=str, required=True, help='path of folders to clean image')
    parser.add_argument('--prompts', type=str, required=True, help='prompts string')
    parser.add_argument('--save_dir', type=str, required=True, help='dir to save evaluation results')
    parser.add_argument('--scene', type=str, default='perturbed_output', required=False, help='choose perturbed_output or original_output')
    parser.add_argument('--scene2', type=str, default='resize_image', required=False, help='the other folder name')
    parser.add_argument('--is_target', type=int, default=0, required=False, help='compare with target images')
    parser.add_argument('--map_path', type=str, default='', required=False, help='path of map json')
    parser.add_argument('--target_path', type=str, default='', required=False, help='path of target images')
    parser.add_argument('--model_name_or_path', type=str, default="./evaluations/my_clip/ViT-B-32.pt", required=True, help='ViT-B/32')
    parser.add_argument('--device', type=str, default='cuda', required=True, help='cuda or cpu')
    parser.add_argument('--input_name', type=str, default='set_B', required=True, help='set_B or none')
    parser.add_argument('--out_out', type=int, default=0, required=False, help='calculate ims between protected output and original output')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    device = args.device if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model_name_or_path, device=device)
    prompts = args.prompts.split(';')
    prompt_paths_list = []
    for i in range(0, len(prompts)):
        prompt_paths_list.append(prompts[i].replace(' ', '_'))
    persons_list = os.listdir(args.data_dirs)
    if args.is_target == 1 and args.map_path != '':
        with open(args.map_path, "r", encoding="utf-8") as f:
            max_dist_dict = json.load(f)
    # 遍历对抗图片列表
    clip_list = [0.0 for _ in range(0, len(persons_list))]
    prompts_ism_list = []
    for i in range(0, len(prompt_paths_list)):
        prompts_ism_list.append([0.0 for _ in range(0, len(persons_list))])
    with torch.no_grad():
        for k, person_id in enumerate(persons_list):
            tmp_clip_list = list()
            if args.is_target == 0: # 原始图片
                list_id_path = os.path.join(args.emb_dirs, person_id, 'set_B')
            elif args.is_target == 1 and args.map_path != '': # 最大mse嵌入损失 target图片
                target_person_id = max_dist_dict[person_id]
                list_id_path = os.path.join(args.emb_dirs, target_person_id, 'set_B')
            elif args.is_target == 1 and args.target_path != '': # 特定target图片
                list_id_path = args.target_path
            # 遍历提示词列表
            # pdb.set_trace()
            for j, prompt_name in enumerate(prompt_paths_list):
                image_path = os.path.join(args.data_dirs, person_id, prompt_name)
                if args.out_out == 1:
                    list_id_path = os.path.join(args.emb_dirs, person_id, prompt_name)
                # 获取id image的平均特征
                ave_id_embedding = 0
                id_img_list = os.listdir(list_id_path)
                for id_img in id_img_list:
                    id_img_path = os.path.join(list_id_path, id_img)
                    image = preprocess(Image.open(id_img_path)).unsqueeze(0).to(device)
                    ave_id_embedding += model.encode_image(image)[0]
                ave_id_embedding /= len(id_img_list)
                ave_embedding = 0
                img_list = os.listdir(image_path)
                for img in img_list:
                    img_path = os.path.join(image_path, img)
                    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                    ave_embedding += model.encode_image(image)[0]
                ave_embedding /= len(img_list)
                # pdb.set_trace()
                ism = F.cosine_similarity(ave_embedding, ave_id_embedding, dim=-1).mean()
                if ism == None:
                    tmp_clip_list.append(0)
                    prompts_ism_list[j][k] = 0
                else:
                    tmp_clip_list.append(ism.item())
                    prompts_ism_list[j][k] = ism.item()
            print("tmp_clip_list:{}".format(str(tmp_clip_list)))
            clip_list[k] = sum(tmp_clip_list) * 1.0 / len(tmp_clip_list)
    print("clip_list:{}".format(clip_list))
    clip_list_data = np.array(clip_list)
    clip_sample_std = np.std(clip_list_data, ddof=1)
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.scene + '_' + args.scene2 + '_' + 'CLIP' + '.txt')
    with open(save_path, 'w') as f:
        f.write('CLIP IMS Mean\n')
        f.write(str(clip_list) + '\n')
        f.write(str(sum(clip_list) * 1.0 / len(clip_list)) + '\n')
        f.write('CLIP IMS STD\n')
        f.write(str(clip_sample_std) + '\n')
        # 写入每个提示词的ISM和FDR
        f.write('CLIP ISM for Each Prompt:\n')
        for j, prompt_name in enumerate(prompt_paths_list):
            f.write(prompt_name + '\n')
            f.write("ISM list: " + str(prompts_ism_list[j]) + "\n")
            f.write("ISM average value: " + str(sum(prompts_ism_list[j]) * 1.0 / len(prompts_ism_list[j])) + "\n")
            prompt_ism_sample_std = np.std(prompts_ism_list[j], ddof=1)
            f.write('prompt_ism_sample_std: ' + str(prompt_ism_sample_std) + '\n')
    return
            
if __name__ == '__main__':
    main()