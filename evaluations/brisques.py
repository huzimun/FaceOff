import argparse
import os
from PIL import Image
from brisque import BRISQUE
import pdb
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Brisque')
    # parser.add_argument('--prompt_path', default=None, help='path to input image folders')
    parser.add_argument('--data_dirs', type=str, default='', required=True, help='path of folders to perturbed image')
    parser.add_argument('--scene', type=str, default='perturbed_output', required=False, help='choose perturbed_output or original_output')
    parser.add_argument('--prompts', type=str, required=True, help='prompts string')
    parser.add_argument('--save_dir', type=str, required=True, help='dir to save evaluation results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    obj = BRISQUE(url=False)
    prompts = args.prompts.split(';')
    prompt_paths_list = list()
    for te1 in range(0, len(prompts)):
        prompt_paths_list.append(prompts[te1].replace(' ', '_'))
    # prompt_paths_list = [prompts[0].replace(' ', '_'), prompts[1].replace(' ', '_')]
    brisque_lists = list()
    for k, prompt_name in enumerate(prompt_paths_list):
        tmp_brisque_list = list()
        persons_dirs = args.data_dirs + prompt_name + '/'
        persons_dirs_list = os.listdir(persons_dirs)
        for persons in persons_dirs_list:
            image_path = persons_dirs + persons + '/' + args.scene + '/'
            prompt_score = 0
            count = 0
            for img_name in os.listdir(image_path):
                if "png" in img_name or "jpg" in img_name:
                    img_path = os.path.join(image_path, img_name)
                    img = Image.open(img_path)
                    brisque_score = obj.score(img)
                    # print(brisque_score)
                    prompt_score += brisque_score
                    count += 1
            tmp_brisque_list.append(prompt_score/count)
        brisque_lists.append(tmp_brisque_list)
    print('brisque_lists: {}'.format(brisque_lists))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = prompt_name
    save_path =  args.save_dir + '/brisque_' + args.scene + '_' + save_path + '.txt'
    print("save_path:{}".format(save_path))
    with open(save_path, 'w') as f:
        f.write(str(brisque_lists) + '\n')
        for k, prompt_name in enumerate(prompt_paths_list):
            if len(brisque_lists[k]) > 0:
                print("For prompt: {}".format(prompt_name))
                f.write(prompt_name + '\n')
                print("brisque_list: {}".format(brisque_lists[k]))
                f.write(str(brisque_lists[k]) + '\n')
                print("average brisque: {}".format(sum(brisque_lists[k]) / len(brisque_lists[k])))
                f.write(str(sum(brisque_lists[k]) / len(brisque_lists[k])) + '\n')
                
    return

if __name__ == '__main__':
    main()