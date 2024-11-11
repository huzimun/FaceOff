from deepface import DeepFace
import numpy as np
import os
import torch
import torch.nn.functional as F
import argparse
from compute_idx_emb import compute_idx_embedding
from datetime import datetime
import time
import pdb
import json

def compute_face_embedding(img_path, model_name):
    """Extract face embedding vector of given image
    Args:
        img_path (str): path to image
    Returns:
        None: no face found
        vector: return the embedding of biggest face among the all found faces
    """
    try:
        resps = DeepFace.represent(img_path = os.path.join(img_path), 
                                   model_name=model_name, # "ArcFace"
                                   enforce_detection=True, 
                                   detector_backend="retinaface", 
                                   align=True)
        if resps == 1:
            # detect only 1 face
            return np.array(resps[0]["embedding"])
        else:
            # detect more than 1 faces, choose the biggest one
            resps = list(resps)
            resps.sort(key=lambda resp: resp["facial_area"]["h"]*resp["facial_area"]["w"], reverse=True)
            return np.array(resps[0]["embedding"])
    except Exception:
        # no face found
        return None

def get_precomputed_embedding(path):
    """Get face embedding by loading the path to numpy file
    Args:
        path (str): path to numpy file 
    Returns:
        vector: face embedding
    """
    return np.load(path)

def matching_score_id(image_path, avg_embedding, model_name):
    """getting the matching score between face image and precomputed embedding

    Args:
        img (2D images): images
        emb (vector): face embedding

    Returns:
        None: cannot detect face from img
        int: identity score matching
    """
    image_emb = compute_face_embedding(img_path=image_path, model_name=model_name)
    id_emb = avg_embedding
    if image_emb is None:
        return None
    image_emb, id_emb = torch.Tensor(image_emb), torch.Tensor(id_emb)
    ism = F.cosine_similarity(image_emb, id_emb, dim=0)
    return ism

def matching_score_genimage_id(images_path, list_id_path, model_name):
    image_list = os.listdir(images_path)
    fail_detection_count = 0
    ave_ism = 0
    avg_embedding = compute_idx_embedding(list_id_path, model_name)

    for image_name in image_list:
        image_path = os.path.join(images_path, image_name)
        ism = matching_score_id(image_path, avg_embedding, model_name)
        if ism is None:
            fail_detection_count += 1
        else:
            ave_ism += ism
    if fail_detection_count != len(image_list):
        return ave_ism/(len(image_list)-fail_detection_count), fail_detection_count/len(image_list)
    return None, 1

def parse_args():
    parser = argparse.ArgumentParser(description='FDFR and ISM evaluation')
    parser.add_argument('--data_dirs', type=str, default='', required=True, help='path of folders to perturbed image')
    parser.add_argument('--emb_dirs', type=str, required=True, help='path of folders to clean image')
    parser.add_argument('--prompts', type=str, required=True, help='prompts string')
    parser.add_argument('--save_dir', type=str, required=True, help='dir to save evaluation results')
    parser.add_argument('--scene', type=str, default='perturbed_output', required=False, help='choose perturbed_output or original_output')
    parser.add_argument('--scene2', type=str, default='resize_image', required=False, help='the other folder name')
    parser.add_argument('--is_target', type=int, default=0, required=False, help='compare with target images')
    parser.add_argument('--map_path', type=str, default='', required=False, help='path of map json')
    parser.add_argument('--target_path', type=str, default='', required=False, help='path of target images')
    parser.add_argument('--model_name', type=str, default='ArcFace', required=True, help='VGG-Face, ArcFace')
    parser.add_argument('--input_name', type=str, default='set_B', required=True, help='set_B or none')
    parser.add_argument('--out_out', type=int, default=0, required=False, help='calculate ims between protected output and original output')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    prompts = args.prompts.split(';')
    prompt_paths_list = []
    for i in range(0, len(prompts)):
        prompt_paths_list.append(prompts[i].replace(' ', '_'))
    persons_list = os.listdir(args.data_dirs)
    if args.is_target == 1 and args.map_path != '':
        with open(args.map_path, "r", encoding="utf-8") as f:
            max_dist_dict = json.load(f)
    # 遍历对抗图片列表
    ism_list = [0.0 for _ in range(0, len(persons_list))]
    fdr_list = [0.0 for _ in range(0, len(persons_list))]
    for k, person_id in enumerate(persons_list):
        tmp_ism_list = list()
        tmp_fdr_list = list()
        if args.is_target == 0: # 原始图片
            list_id_path = os.path.join(args.emb_dirs, person_id+'/'+args.input_name)
        elif args.is_target == 1 and args.map_path != '': # 最大mse嵌入损失 target图片
            target_person_id = max_dist_dict[person_id]
            list_id_path = os.path.join(args.emb_dirs, target_person_id+'/'+args.input_name)
        elif args.is_target == 1 and args.target_path != '': # 特定target图片
            list_id_path = args.target_path
        # 遍历提示词列表
        for prompt_name in prompt_paths_list:
            image_path = os.path.join(args.data_dirs, person_id, prompt_name)
            if args.out_out == 1:
                list_id_path = os.path.join(args.emb_dirs, person_id, prompt_name)
            ism, fdr = matching_score_genimage_id(image_path, list_id_path, args.model_name)
            if ism == None:
                tmp_ism_list.append(0)
            else:
                tmp_ism_list.append(ism.item())
            tmp_fdr_list.append(fdr)
        print("tmp_ism_list:{}".format(str(tmp_ism_list)))
        print("tmp_fdr_list:{}".format(str(tmp_fdr_list)))
        ism_list[k] = sum(tmp_ism_list) * 1.0 / len(tmp_ism_list)
        fdr_list[k] = sum(tmp_fdr_list) * 1.0 / len(tmp_fdr_list)
    print("ism_list:{}".format(ism_list))
    ism_list_data = np.array(ism_list)
    ism_sample_std = np.std(ism_list_data, ddof=1)
    print("fdr_list:{}".format(fdr_list))
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.scene + '_' + args.scene2 + '_' + args.model_name + '.txt')
    with open(save_path, 'w') as f:
        f.write('FDR Mean\n')
        f.write(str(fdr_list) + '\n')
        f.write(str(sum(fdr_list) * 1.0 / len(fdr_list)) + '\n')
        f.write('ISM Mean\n')
        f.write(str(ism_list) + '\n')
        f.write(str(sum(ism_list) * 1.0 / len(ism_list)) + '\n')
        f.write('ISM sample std\n')
        f.write(str(ism_sample_std) + '\n')
    return
            
if __name__ == '__main__':
    main()

