export adversarial_folder_name="VGGFace2_IP-Adapter"
echo $adversarial_folder_name
export device="cuda:0"
export adversarial_input_dir="./outputs/adversarial_images/VGGFace2"
export customization_output_dir="./outputs/customization_outputs/${adversarial_folder_name}"
export evaluation_output_dir="./outputs/evaluation_outputs/${adversarial_folder_name}"
export prompts="a_photo_of_person;a_dslr_portrait_of_person"
export VGGFace2="/data1/humw/Datasets/VGGFace2"
export clip_model_name_or_path="/data1/humw/Pretrains/ViT-B-32.pt"
echo $prompts

export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/eval/2_eval_new_ip-adapter.sh" $save_config_dir

# CLIP
python3 ./evaluations/my_clip/my_clip.py \
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
    --input_name "set_B" \
    --out_out 0

export adversarial_folder_name="sdxl-VGGFace2_IP-Adapter"
echo $adversarial_folder_name
export device="cuda:0"
export adversarial_input_dir="./outputs/adversarial_images/VGGFace2"
export customization_output_dir="./outputs/customization_outputs/${adversarial_folder_name}"
export evaluation_output_dir="./outputs/evaluation_outputs/${adversarial_folder_name}"
export prompts="a_photo_of_person;a_dslr_portrait_of_person"
export VGGFace2="/data1/humw/Datasets/VGGFace2"
export clip_model_name_or_path="/data1/humw/Pretrains/ViT-B-32.pt"
echo $prompts

export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/eval/2_eval_new_ip-adapter.sh" $save_config_dir

# CLIP
python3 ./evaluations/my_clip/my_clip.py \
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
    --input_name "set_B" \
    --out_out 0

export adversarial_folder_name="sdsT5_eps16_steps100_gmode-"
echo $adversarial_folder_name
export device="cuda:0"
export adversarial_input_dir="./outputs/adversarial_images/${adversarial_folder_name}"
export customization_output_dir="./outputs/customization_outputs/${adversarial_folder_name}"
export evaluation_output_dir="./outputs/evaluation_outputs/${adversarial_folder_name}"
export prompts="a_photo_of_person;a_dslr_portrait_of_person"
export VGGFace2="/data1/humw/Datasets/VGGFace2"
export clip_model_name_or_path="/data1/humw/Pretrains/ViT-B-32.pt"
echo $prompts

export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/eval/2_eval_new_ip-adapter.sh" $save_config_dir

# CLIP
python3 ./evaluations/my_clip/my_clip.py \
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

export adversarial_folder_name="sdxl-sdsT5_eps16_steps100_gmode-"
echo $adversarial_folder_name
export device="cuda:0"
export adversarial_input_dir="./outputs/adversarial_images/${adversarial_folder_name}"
export customization_output_dir="./outputs/customization_outputs/${adversarial_folder_name}"
export evaluation_output_dir="./outputs/evaluation_outputs/${adversarial_folder_name}"
export prompts="a_photo_of_person;a_dslr_portrait_of_person"
export VGGFace2="/data1/humw/Datasets/VGGFace2"
export clip_model_name_or_path="/data1/humw/Pretrains/ViT-B-32.pt"
echo $prompts

export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/eval/2_eval_new_ip-adapter.sh" $save_config_dir

# CLIP
python3 ./evaluations/my_clip/my_clip.py \
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
