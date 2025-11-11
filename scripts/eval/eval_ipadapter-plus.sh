export adversarial_folder_name="ASPL_ace_VGGFace2_SD15_mist"
export dataset="./outputs/adversarial_images/VGGFace2"
export map_json_path="./max_clip_cosine_distance_map_VGGFace2.json"
export customization_folder_name="ipadapter-plus_sdxl_${adversarial_folder_name}"
export customization_output_dir="./outputs/customization_outputs/${customization_folder_name}"
export evaluation_output_dir="./outputs/evaluation_outputs/${customization_folder_name}"
export prompts="a_photo_of_person;a_dslr_portrait_of_person"
echo $prompts

export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/eval_ipadapter_ipadapter-plus.sh" $save_config_dir

# original output and original input
# ArcFace
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $customization_output_dir \
    --emb_dirs $dataset \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --scene2 "original_input" \
    --is_target 0 \
    --map_path $map_json_path \
    --target_path "" \
    --model_name "ArcFace" \
    --input_name "set_B" \
    --out_out 0

# export adversarial_folder_name="ASPL_ace_new-CelebA-HQ_SD15_mist"
# export dataset="./outputs/adversarial_images/new-CelebA-HQ"
# export map_json_path="./max_clip_cosine_distance_map_new-CelebA-HQ.json"
# export customization_folder_name="ipadapter-plus_sdxl_${adversarial_folder_name}"
# export customization_output_dir="./outputs/customization_outputs/${customization_folder_name}"
# export evaluation_output_dir="./outputs/evaluation_outputs/${customization_folder_name}"
# export prompts="a_photo_of_person;a_dslr_portrait_of_person"
# echo $prompts

# export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
# mkdir $save_config_dir
# cp "./scripts/eval_ipadapter_ipadapter-plus.sh" $save_config_dir

# # original output and original input
# # ArcFace
# python ./evaluations/ism_fdfr.py \
#     --prompts $prompts \
#     --data_dir $customization_output_dir \
#     --emb_dirs $dataset \
#     --save_dir $evaluation_output_dir \
#     --scene "protected_output" \
#     --scene2 "original_input" \
#     --is_target 0 \
#     --map_path $map_json_path \
#     --target_path "" \
#     --model_name "ArcFace" \
#     --input_name "set_B" \
#     --out_out 0
