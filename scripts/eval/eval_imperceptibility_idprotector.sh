export VGGFace2_adversarial_folder_name="idprotector_VGGFace2_224"
export adversarial_folder_name="${VGGFace2_adversarial_folder_name}"
export customization_folder_name="photomaker_sdxl_${adversarial_folder_name}"
export customization_output_dir="./outputs/customization_outputs/${customization_folder_name}"
export evaluation_output_dir="./outputs/evaluation_outputs/${customization_folder_name}"
export map_json_path="./max_clip_cosine_distance_map_VGGFace2.json"
export dataset="/home/humw/Datasets/VGGFace2_224"
export prompts="a_photo_of_sks_person;a_dslr_portrait_of_sks_person"
export device="cuda:7"
echo $prompts

export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/eval/eval_imperceptibility_idprotector.sh" $save_config_dir

# protected_input and original_input: FID, LPIPS, SSIM, PSNR
# protected_input: LIQE, BRISQUE
python3 ./evaluations/pyiqa/iqa_metric_idprotector.py \
    --data_dirs $adversarial_folder_name \
    --emb_dirs $dataset \
    --save_dir $evaluation_output_dir \
    --sub_folder '' \
    --scene "protected_input" \
    --scene2 "original_input" \
    --device $device


export adversarial_folder_name="idprotector_new-CelebA-HQ_224"
export customization_folder_name="photomaker_sdxl_${adversarial_folder_name}"
export customization_output_dir="./outputs/customization_outputs/${customization_folder_name}"
export evaluation_output_dir="./outputs/evaluation_outputs/${customization_folder_name}"
export map_json_path="./max_clip_cosine_distance_map_new-CelebA-HQ.json"
export dataset="/home/humw/Datasets/new-CelebA-HQ_224"
export prompts="a_photo_of_sks_person;a_dslr_portrait_of_sks_person"
export device="cuda:7"
echo $prompts

export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/eval/eval_imperceptibility_idprotector.sh" $save_config_dir

# protected_input and original_input: FID, LPIPS, SSIM, PSNR
# protected_input: LIQE, BRISQUE
python3 ./evaluations/pyiqa/iqa_metric_idprotector.py \
    --data_dirs $adversarial_folder_name \
    --emb_dirs $dataset \
    --save_dir $evaluation_output_dir \
    --sub_folder '' \
    --scene "protected_input" \
    --scene2 "original_input" \
    --device $device
