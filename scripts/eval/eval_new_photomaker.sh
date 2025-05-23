export adversarial_folder_name="photomaker_SDXL-BASE-1_Encoder_attack_conda-photomaker_new-CelebA-HQ_vae15-ipadapter-photomaker_mix_eot-0_yingbu_agm-2_norm-0_lora-0"
echo $adversarial_folder_name
export device="cuda:1"
export adversarial_input_dir="./outputs/adversarial_images/${adversarial_folder_name}"
export customization_output_dir="./outputs/customization_outputs/${adversarial_folder_name}"
export evaluation_output_dir="./outputs/evaluation_outputs/${adversarial_folder_name}"
export prompts="a_photo_of_sks_person;a_dslr_portrait_of_sks_person"
export Dataset="/data1/humw/Datasets/new-CelebA-HQ"
export clip_model_name_or_path="/data1/humw/Pretrains/ViT-B-32.pt"
echo $prompts

export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/eval/eval_new_ip-adapter.sh" $save_config_dir

# IMS: protected output and original input
# ArcFace
python3 ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $customization_output_dir \
    --emb_dirs $Dataset \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --scene2 "original_input" \
    --is_target 0 \
    --map_path "" \
    --target_path "" \
    --model_name "ArcFace" \
    --input_name "set_B" \
    --out_out 0

# # VGG-Face
# python3 ./evaluations/ism_fdfr.py \
#     --prompts $prompts \
#     --data_dir $customization_output_dir \
#     --emb_dirs $Dataset \
#     --save_dir $evaluation_output_dir \
#     --scene "protected_output" \
#     --scene2 "original_input" \
#     --is_target 0 \
#     --map_path "" \
#     --target_path "" \
#     --model_name "VGG-Face" \
#     --input_name "set_B" \
#     --out_out 0

# # CLIP
# python3 ./evaluations/my_clip/my_clip.py \
#     --prompts $prompts \
#     --data_dir $customization_output_dir \
#     --emb_dirs $Dataset \
#     --save_dir $evaluation_output_dir \
#     --scene "protected_output" \
#     --scene2 "original_input" \
#     --is_target 0 \
#     --map_path "" \
#     --target_path "" \
#     --model_name_or_path $clip_model_name_or_path \
#     --device $device \
#     --input_name "set_B" \
#     --out_out 0

# # IQA: protected output and original input
# # FID (LIQE, BRISQUE没测）
# python3 ./evaluations/pyiqa/iqa_metric_for_output.py \
#     --data_dir $customization_output_dir \
#     --emb_dir $Dataset \
#     --prompts $prompts \
#     --save_dir $evaluation_output_dir \
#     --scene "protected_output" \
#     --scene2 "original_input" \
#     --device $device

# # protected_input and original_input: FID, LPIPS, SSIM, PSNR
# # protected_input: LIQE, BRISQUE
# python3 ./evaluations/pyiqa/iqa_metric.py \
#     --data_dir $adversarial_input_dir \
#     --emb_dirs $Dataset \
#     --save_dir $evaluation_output_dir \
#     --sub_folder '' \
#     --scene "protected_input" \
#     --scene2 "original_input" \
#     --device $device
