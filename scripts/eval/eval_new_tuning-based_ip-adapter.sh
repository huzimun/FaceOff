export adversarial_folder_name="ASPL_SD15"
echo $adversarial_folder_name
export device="cuda:0"
export adversarial_input_dir="./outputs/adversarial_images/${adversarial_folder_name}"
export customization_output_dir="./outputs/customization_outputs/ASPL_SD15_ipadapter"
export evaluation_output_dir="./outputs/evaluation_outputs/ASPL_SD15_ipadapter"
export original_output_dir="./outputs/customization_outputs/VGGFace2_IP-Adapter"
export prompts="a_photo_of_person;a_dslr_portrait_of_person"
export VGGFace2="/data1/humw/Datasets/VGGFace2"
export clip_model_name_or_path="/data1/humw/Codes/Anti-DreamBooth/evaluations/my_clip/ViT-B-32.pt"
echo $prompts

export save_config_dir="./outputs/config_scripts_logs/ASPL_SD15_ipadapter"
mkdir $save_config_dir
cp "./scripts/eval/eval_new_tuning-based_ip-adapter.sh" $save_config_dir

# IMS: protected output and original input
# ArcFace
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $customization_output_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --scene2 "original_input" \
    --is_target 0 \
    --map_path "" \
    --target_path "" \
    --model_name "ArcFace" \
    --input_name "set_B" \
    --out_out 0
# VGG-Face
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $customization_output_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --scene2 "original_input" \
    --is_target 0 \
    --map_path "" \
    --target_path "" \
    --model_name "VGG-Face" \
    --input_name "set_B" \
    --out_out 0
# CLIP
python ./evaluations/my_clip/my_clip.py \
    --prompts $prompts \
    --data_dir $customization_output_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --scene2 "original_input" \
    --is_target 0 \
    --map_path "" \
    --target_path "" \
    --model_name_or_path $clip_model_name_or_path \
    --device $device \
    --input_name "" \
    --out_out 0

# # IMS: original output and original input
# # ArcFace
# python ./evaluations/ism_fdfr.py \
#     --prompts $prompts \
#     --data_dir $original_output_dir \
#     --emb_dirs $VGGFace2 \
#     --save_dir $evaluation_output_dir \
#     --scene "original_output" \
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
#     --data_dir $original_output_dir \
#     --emb_dirs $VGGFace2 \
#     --save_dir $evaluation_output_dir \
#     --scene "original_output" \
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
#     --data_dir $original_output_dir \
#     --emb_dirs $VGGFace2 \
#     --save_dir $evaluation_output_dir \
#     --scene "original_output" \
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
    --sub_folder '' \
    --scene "protected_input" \
    --scene2 "original_input" \
    --device $device

export adversarial_folder_name="CAAT_SD15"
echo $adversarial_folder_name
export device="cuda:0"
export adversarial_input_dir="./outputs/adversarial_images/${adversarial_folder_name}"
export customization_output_dir="./outputs/customization_outputs/CAAT_SD15_ipadapter"
export evaluation_output_dir="./outputs/evaluation_outputs/CAAT_SD15_ipadapter"
export original_output_dir="./outputs/customization_outputs/VGGFace2_IP-Adapter"
export prompts="a_photo_of_person;a_dslr_portrait_of_person"
export VGGFace2="/data1/humw/Datasets/VGGFace2"
export clip_model_name_or_path="/data1/humw/Codes/Anti-DreamBooth/evaluations/my_clip/ViT-B-32.pt"
echo $prompts

export save_config_dir="./outputs/config_scripts_logs/CAAT_SD15_ipadapter"
mkdir $save_config_dir
cp "./scripts/eval/eval_new_tuning-based_ip-adapter.sh" $save_config_dir

# IMS: protected output and original input
# ArcFace
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $customization_output_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --scene2 "original_input" \
    --is_target 0 \
    --map_path "" \
    --target_path "" \
    --model_name "ArcFace" \
    --input_name "set_B" \
    --out_out 0
# VGG-Face
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $customization_output_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --scene2 "original_input" \
    --is_target 0 \
    --map_path "" \
    --target_path "" \
    --model_name "VGG-Face" \
    --input_name "set_B" \
    --out_out 0
# CLIP
python ./evaluations/my_clip/my_clip.py \
    --prompts $prompts \
    --data_dir $customization_output_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --scene2 "original_input" \
    --is_target 0 \
    --map_path "" \
    --target_path "" \
    --model_name_or_path $clip_model_name_or_path \
    --device $device \
    --input_name "" \
    --out_out 0

# # IMS: original output and original input
# # ArcFace
# python ./evaluations/ism_fdfr.py \
#     --prompts $prompts \
#     --data_dir $original_output_dir \
#     --emb_dirs $VGGFace2 \
#     --save_dir $evaluation_output_dir \
#     --scene "original_output" \
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
#     --data_dir $original_output_dir \
#     --emb_dirs $VGGFace2 \
#     --save_dir $evaluation_output_dir \
#     --scene "original_output" \
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
#     --data_dir $original_output_dir \
#     --emb_dirs $VGGFace2 \
#     --save_dir $evaluation_output_dir \
#     --scene "original_output" \
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
    --sub_folder '' \
    --scene "protected_input" \
    --scene2 "original_input" \
    --device $device
