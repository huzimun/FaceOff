import pyiqa
import torch
import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Image Quality Evaluation')
    parser.add_argument('--data_dirs', type=str, default='', required=True, help='path of folders to perturbed image')
    parser.add_argument('--emb_dirs', type=str, required=True, help='path of folders to clean image')
    parser.add_argument('--save_dir', type=str, required=True, help='dir to save evaluation results')
    parser.add_argument('--sub_folder', type=str, default='', required=False, help='set_B or none')
    parser.add_argument('--scene', type=str, default='perturbed_input', required=False, help='choose perturbed_output or original_output')
    parser.add_argument('--scene2', type=str, default='original_input', required=False, help='the other folder name')
    parser.add_argument('--device', type=str, default='cuda', required=True, help='cuda or cpu')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    device = args.device if torch.cuda.is_available() else "cpu"

    # FR metrics
    # LPIPS↓
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    # FID↓
    fid_metric = pyiqa.create_metric('fid', device=device)
    # SSIM↑
    ssim_metric = pyiqa.create_metric('ssimc', device=device)
    # PSNR↑
    psnr_metric = pyiqa.create_metric('psnr', device=device)

    # NR metrics
    # LIQE↑
    liqe_metric = pyiqa.create_metric('liqe', device=device)
    # BRISQUE↑
    brisque_metric = pyiqa.create_metric('brisque', device=device)

    persons_list = os.listdir(args.data_dirs)
    
    lpips_list = [0.0 for _ in range(0, len(persons_list))]
    fid_list = [0.0 for _ in range(0, len(persons_list))]
    ssim_list = [0.0 for _ in range(0, len(persons_list))]
    psnr_list = [0.0 for _ in range(0, len(persons_list))]
    liqe_list = [0.0 for _ in range(0, len(persons_list))]
    brisque_list = [0.0 for _ in range(0, len(persons_list))]
    with torch.no_grad():
        for k, person_id in enumerate(persons_list):
            list_id_path = os.path.join(args.emb_dirs, person_id, 'set_B')
            image_path = os.path.join(args.data_dirs, person_id + '/' + args.sub_folder)

            ave_lpips_score = 0
            ave_ssim_score = 0
            ave_psnr_score = 0
            ave_liqe_score = 0
            ave_brisque_score = 0
            id_img_list = sorted(os.listdir(list_id_path)) # 一定要sort，防止因为不同的前后缀导致的差错
            img_list = sorted(os.listdir(image_path))
            for idx, id_img in enumerate(id_img_list):
                id_img_path = os.path.join(list_id_path, id_img)
                img_path = os.path.join(image_path, img_list[idx])

                # FR metrics
                lpips_score = lpips_metric(img_path, id_img_path).item()
                print("lpips_score:{}".format(lpips_score))
                ave_lpips_score += lpips_score
                ssim_score = ssim_metric(img_path, id_img_path).item()
                print("ssim_score:{}".format(ssim_score))
                ave_ssim_score += ssim_score
                psnr_score = psnr_metric(img_path, id_img_path).item()
                print("psnr_score:{}".format(psnr_score))
                ave_psnr_score += psnr_score
                
                # NR metrics
                liqe_score = liqe_metric(img_path).item()
                print("liqe_score:{}".format(liqe_score))
                ave_liqe_score += liqe_score
                brisque_score = brisque_metric(img_path).item()
                print("brisque_score:{}".format(brisque_score))
                ave_brisque_score += brisque_score
            ave_fid_score = fid_metric(image_path, list_id_path).item()
            print("ave_fid_score:{}".format(ave_fid_score))
            fid_list[k] = ave_fid_score
            ave_lpips_score = ave_lpips_score / len(id_img_list)
            ave_ssim_score = ave_ssim_score / len(id_img_list)
            ave_psnr_score = ave_psnr_score / len(id_img_list)
            ave_liqe_score = ave_liqe_score / len(id_img_list)
            ave_brisque_score = ave_brisque_score / len(id_img_list)
            print("ave_lpips_score:{}".format(ave_lpips_score))
            print("ave_ssim_score:{}".format(ave_ssim_score))
            print("ave_psnr_score:{}".format(ave_psnr_score))
            print("ave_liqe_score:{}".format(ave_liqe_score))
            print("ave_brisque_score:{}".format(ave_brisque_score))
            lpips_list[k] = ave_lpips_score
            ssim_list[k] = ave_ssim_score
            psnr_list[k] = ave_psnr_score
            liqe_list[k] = ave_liqe_score
            brisque_list[k] = ave_brisque_score
    print("fid_list:{}".format(fid_list))
    print("lpips_list:{}".format(lpips_list))
    print("ssim_list:{}".format(ssim_list))
    print("psnr_list:{}".format(psnr_list))
    print("liqe_list:{}".format(liqe_list))
    print("brisque_list:{}".format(brisque_list))

    fid_list_data = np.array(fid_list)
    fid_sample_std = np.std(fid_list_data, ddof=1)
    lpips_list_data = np.array(lpips_list)
    lpips_sample_std = np.std(lpips_list_data, ddof=1)
    ssim_list_data = np.array(ssim_list)
    ssim_sample_std = np.std(ssim_list_data, ddof=1)
    psnr_list_data = np.array(psnr_list)
    psnr_sample_std = np.std(psnr_list_data, ddof=1)
    liqe_list_data = np.array(liqe_list)
    liqe_sample_std = np.std(liqe_list_data, ddof=1)
    brisque_list_data = np.array(brisque_list)
    brisque_sample_std = np.std(brisque_list_data, ddof=1)
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.scene + '_' + args.scene2 + '_' + 'FID' + '.txt')
    with open(save_path, 'w') as f:
        f.write('mean\n')
        f.write(str(fid_list) + '\n')
        f.write(str(sum(fid_list) * 1.0 / len(fid_list)) + '\n')
        f.write('sample std\n')
        f.write(str(fid_sample_std) + '\n')
    save_path = os.path.join(args.save_dir, args.scene + '_' + args.scene2 + '_' + 'LPIPS' + '.txt')
    with open(save_path, 'w') as f:
        f.write('mean\n')
        f.write(str(lpips_list) + '\n')
        f.write(str(sum(lpips_list) * 1.0 / len(lpips_list)) + '\n')
        f.write('sample std\n')
        f.write(str(lpips_sample_std) + '\n')
    save_path = os.path.join(args.save_dir, args.scene + '_' + args.scene2 + '_' + 'SSIM' + '.txt')
    with open(save_path, 'w') as f:
        f.write('mean\n')
        f.write(str(ssim_list) + '\n')
        f.write(str(sum(ssim_list) * 1.0 / len(ssim_list)) + '\n')
        f.write('sample std\n')
        f.write(str(ssim_sample_std) + '\n')
    save_path = os.path.join(args.save_dir, args.scene + '_' + args.scene2 + '_' + 'PSNR' + '.txt')
    with open(save_path, 'w') as f:
        f.write('mean\n')
        f.write(str(psnr_list) + '\n')
        f.write(str(sum(psnr_list) * 1.0 / len(psnr_list)) + '\n')
        f.write('sample std\n')
        f.write(str(psnr_sample_std) + '\n')
    save_path = os.path.join(args.save_dir, args.scene + '_'  + 'LIQE_for_input' + '.txt')
    with open(save_path, 'w') as f:
        f.write('mean\n')
        f.write(str(liqe_list) + '\n')
        f.write(str(sum(liqe_list) * 1.0 / len(liqe_list)) + '\n')
        f.write('sample std\n')
        f.write(str(liqe_sample_std) + '\n')
    save_path = os.path.join(args.save_dir, args.scene + '_'  + 'BRISQUE_for_input' + '.txt')
    with open(save_path, 'w') as f:
        f.write('mean\n')
        f.write(str(brisque_list) + '\n')
        f.write(str(sum(brisque_list) * 1.0 / len(brisque_list)) + '\n')
        f.write('sample std\n')
        f.write(str(brisque_sample_std) + '\n')
    return
            
if __name__ == '__main__':
    main()