export adversarial_folder_name="VGGFace2"
echo $adversarial_folder_name
export device="cuda:0"
export adversarial_input_dir="./outputs/adversarial_images/${adversarial_folder_name}"
export customization_output_dir="./outputs/customization_outputs/${adversarial_folder_name}"
export evaluation_output_dir="./outputs/evaluation_outputs/${adversarial_folder_name}"
export original_output_dir="./outputs/customization_outputs/VGGFace2"
export prompts="a_photo_of_sks_person;a_dslr_portrait_of_sks_person"
export VGGFace2="/data1/humw/Datasets/VGGFace2"
export clip_model_name_or_path="/data1/humw/Codes/Anti-DreamBooth/evaluations/my_clip/ViT-B-32.pt"
echo $prompts

export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/eval/eval_photomaker_new_FaceOff.sh" $save_config_dir

# # IMS: protected output and original input
# # ArcFace
# python ./evaluations/ism_fdfr.py \
#     --prompts $prompts \
#     --data_dir $customization_output_dir \
#     --emb_dirs $VGGFace2 \
#     --save_dir $evaluation_output_dir \
#     --scene "protected_output" \
#     --scene2 "original_input" \
#     --is_target 0 \
#     --map_path "" \
#     --target_path "" \
#     --model_name "ArcFace" \
#     --input_name "set_B" \
#     --out_out 0
# # VGG-Face
# python ./evaluations/ism_fdfr.py \
#     --prompts $prompts \
#     --data_dir $customization_output_dir \
#     --emb_dirs $VGGFace2 \
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
# python ./evaluations/my_clip/my_clip.py \
#     --prompts $prompts \
#     --data_dir $customization_output_dir \
#     --emb_dirs $VGGFace2 \
#     --save_dir $evaluation_output_dir \
#     --scene "protected_output" \
#     --scene2 "original_input" \
#     --is_target 0 \
#     --map_path "" \
#     --target_path "" \
#     --model_name_or_path $clip_model_name_or_path \
#     --device $device \
#     --input_name "" \
#     --out_out 0

# protected_input and original_input: FID, LPIPS, SSIM, PSNR
# protected_input: LIQE, BRISQUE
python ./evaluations/pyiqa/iqa_metric.py \
    --data_dir $adversarial_input_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --sub_folder 'set_B' \
    --scene "original_input" \
    --scene2 "original_input" \
    --device $device

# # protected_input and original_input: FID, LPIPS, SSIM, PSNR
# # protected_input: LIQE, BRISQUE
# python ./evaluations/pyiqa/iqa_metric.py \
#     --data_dir $adversarial_input_dir \
#     --emb_dirs $VGGFace2 \
#     --save_dir $evaluation_output_dir \
#     --sub_folder '' \
#     --scene "protected_input" \
#     --scene2 "original_input" \
#     --device $device
