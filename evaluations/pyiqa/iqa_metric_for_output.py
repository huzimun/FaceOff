import pyiqa
import torch
import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Image Quality Evaluation for Customized Images')
    parser.add_argument('--data_dirs', type=str, default='', required=True, help='path of folders to customized images')
    parser.add_argument('--emb_dirs', type=str, required=True, help='path of folders to clean images')
    parser.add_argument('--prompts', type=str, required=True, help='prompts string')
    parser.add_argument('--save_dir', type=str, required=True, help='dir to save evaluation results')
    parser.add_argument('--scene', type=str, default='customized_output', required=False, help='customized_output')
    parser.add_argument('--scene2', type=str, default='original_input', required=False, help='original_input')
    parser.add_argument('--device', type=str, default='cuda', required=True, help='cuda or cpu')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    prompts = args.prompts.split(';')
    prompt_paths_list = []
    for i in range(0, len(prompts)):
        prompt_paths_list.append(prompts[i].replace(' ', '_').replace('<', '').replace('>', ''))
    device = args.device if torch.cuda.is_available() else "cpu"

    # FR metrics
    # FID↓
    fid_metric = pyiqa.create_metric('fid', device=device)

    # NR metrics
    # LIQE↑
    # liqe_metric = pyiqa.create_metric('liqe', device=device)
    # BRISQUE↑
    # brisque_metric = pyiqa.create_metric('brisque', device=device)

    persons_list = os.listdir(args.data_dirs)
    
    fid_list = [0.0 for _ in range(0, len(persons_list))]
    # liqe_list = [0.0 for _ in range(0, len(persons_list))]
    # brisque_list = [0.0 for _ in range(0, len(persons_list))]
    
    prompts_fid_list = []
    # prompts_liqe_list = []
    # prompts_brisque_list = []
    for i in range(0, len(prompt_paths_list)):
        prompts_fid_list.append([0.0 for _ in range(0, len(persons_list))])
        # prompts_liqe_list.append([0.0 for _ in range(0, len(persons_list))])
        # prompts_brisque_list.append([0.0 for _ in range(0, len(persons_list))])
        
    with torch.no_grad():
        for k, person_id in enumerate(persons_list):
            list_id_path = os.path.join(args.emb_dirs, person_id, 'set_B') # 干净图像路径
            image_path = os.path.join(args.data_dirs, person_id) # 定制图像路径

            tmp_fid_list = list()
            # tmp_liqe_list = list()
            # tmp_brisque_list = list()
            id_img_list = sorted(os.listdir(list_id_path))
            # 对每个提示词分别计算指标
            for j, prompt_name in enumerate(prompt_paths_list):
                print("prompt_name:{}".format(prompt_name))
                prompt_img_path = os.path.join(image_path, prompt_name)
                
                # id_liqe_score = 0.0
                # id_brisque_score = 0.0
                img_list = sorted(os.listdir(prompt_img_path))
                for idx in range(0, len(img_list)):
                    img_path = os.path.join(prompt_img_path, img_list[idx])
                    
                    # liqe_score = liqe_metric(img_path).item()
                    # print("liqe_score:{}".format(liqe_score))
                    # id_liqe_score  += liqe_score
                    
                    # brisque_score = brisque_metric(img_path).item()
                    # print("brisque_score:{}".format(brisque_score))
                    # id_brisque_score += brisque_score
                
                # id_liqe_score = id_liqe_score * 1.0 / len(img_list)
                # id_brisque_score = id_brisque_score * 1.0 / len(img_list)
                # tmp_liqe_list.append(id_liqe_score)
                # tmp_brisque_list.append(id_brisque_score)
                # prompts_liqe_list[j][k] = id_liqe_score
                # prompts_brisque_list[j][k] = id_brisque_score
                
                fid_score = fid_metric(prompt_img_path, list_id_path).item()
                print("ave_fid_score:{}".format(fid_score))
                tmp_fid_list.append(fid_score)
                prompts_fid_list[j][k] = fid_score
                
            ave_prompt_fid_score = sum(tmp_fid_list) / len(tmp_fid_list)
            # ave_prompt_liqe_score = sum(tmp_liqe_list) / len(tmp_liqe_list)
            # ave_prompt_brisque_score = sum(tmp_brisque_list) / len(tmp_brisque_list)
            fid_list[k] = ave_prompt_fid_score
            # liqe_list[k] = ave_prompt_liqe_score
            # brisque_list[k] = ave_prompt_brisque_score
    # ave_liqe_score = sum(liqe_list) * 1.0 / len(persons_list)
    # ave_brisque_score = sum(brisque_list) * 1.0 / len(persons_list)
    ave_fid_score = sum(fid_list) * 1.0 / len(persons_list)
    # print("ave_liqe_score:{}".format(ave_liqe_score))
    # print("ave_brisque_score:{}".format(ave_brisque_score))
    print("ave_fid_score:{}".format(ave_fid_score))
    print("fid_list:{}".format(fid_list))
    # print("liqe_list:{}".format(liqe_list))
    # print("brisque_list:{}".format(brisque_list))

    fid_list_data = np.array(fid_list)
    fid_sample_std = np.std(fid_list_data, ddof=1)
    # liqe_list_data = np.array(liqe_list)
    # liqe_sample_std = np.std(liqe_list_data, ddof=1)
    # brisque_list_data = np.array(brisque_list)
    # brisque_sample_std = np.std(brisque_list_data, ddof=1)
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.scene + '_' + args.scene2 + '_' + 'FID' + '.txt')
    with open(save_path, 'w') as f:
        f.write('mean\n')
        f.write(str(fid_list) + '\n')
        f.write(str(ave_fid_score) + '\n')
        f.write('sample std\n')
        f.write(str(fid_sample_std) + '\n')
        # 每个提示词对应的FID列表，均值和样本标准差
        for j in range(len(prompt_paths_list)):
            f.write(prompt_paths_list[j] + '\n')
            f.write(str(prompts_fid_list[j]) + '\n')
            f.write(str(sum(prompts_fid_list[j]) * 1.0 / len(prompts_fid_list[j])) + '\n')
            f.write(str(np.std(prompts_fid_list[j], ddof=1)) + '\n')
    # save_path = os.path.join(args.save_dir, args.scene + '_'  + 'LIQE_for_input' + '.txt')
    # with open(save_path, 'w') as f:
    #     f.write('mean\n')
    #     f.write(str(liqe_list) + '\n')
    #     f.write(str(ave_liqe_score) + '\n')
    #     f.write('sample std\n')
    #     f.write(str(liqe_sample_std) + '\n')
    #     # 每个提示词对应的LIQE列表，均值和样本标准差
    #     for j in range(len(prompt_paths_list)):
    #         f.write(prompt_paths_list[j] + '\n')
    #         f.write(str(prompts_liqe_list[j]) + '\n')
    #         f.write(str(sum(prompts_liqe_list[j]) * 1.0 / len(prompts_liqe_list[j])) + '\n')
    #         f.write(str(np.std(prompts_liqe_list[j], ddof=1)) + '\n')
    # save_path = os.path.join(args.save_dir, args.scene + '_'  + 'BRISQUE_for_input' + '.txt')
    # with open(save_path, 'w') as f:
    #     f.write('mean\n')
    #     # f.write(str(brisque_list) + '\n')
    #     # f.write(str(ave_brisque_score) + '\n')
    #     f.write('sample std\n')
    #     # f.write(str(brisque_sample_std) + '\n')
    #     # 每个提示词对应的BRISQUE列表，均值和样本标准差
    #     for j in range(len(prompt_paths_list)):
    #         f.write(prompt_paths_list[j] + '\n')
    #         # f.write(str(prompts_brisque_list[j]) + '\n')
    #         # f.write(str(sum(prompts_brisque_list[j]) * 1.0 / len(prompts_brisque_list[j])) + '\n')
    #         # f.write(str(np.std(prompts_brisque_list[j], ddof=1)) + '\n')
    return
            
if __name__ == '__main__':
    main()