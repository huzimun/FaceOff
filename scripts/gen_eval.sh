export model_type='photomaker_clip' # 攻击的模型，photomaker_clip，vae，clip，ipadapter
export data_dir_name="mini-VGGFace2" # 输入数据集
export loss_type='d-x' # 损失函数类型，x普通的target损失函数，n（原来错误的n)，d-x是更新后正确的n，d偏离损失函数
export attack_num=100 # 攻击轮次
export alpha=6 # 步长
export eps=16 # 最大噪声阈值
export min_eps=8 # refiner的最小噪声阈值
export input_size=512 # 输入图片的尺寸
export model_input_size=224 # 模型输入图片尺寸
export target_type="max" # target图片的类型，max代表最大clip特征mse距离
export strong_edge=200 # 边缘检测器的强边
export weak_edge=100 # 边缘检测器的弱边
export mean_filter_size=3 # 均值滤波器的尺寸
export update_interval=10 # 阈值更新间隔
export noise_budget_refiner=1 # 是否使用refiner
export device="cuda:1"
export adversarial_folder_name="metacloak_noise-16_min-10"
echo $adversarial_folder_name
export adversarial_input_dir="./output/photomaker/adversarial_images/${adversarial_folder_name}"
export customization_output_dir="./output/photomaker/customization_outputs/${adversarial_folder_name}"
export evaluation_output_dir="./output/photomaker/evaluation_outputs/${adversarial_folder_name}"
export original_output_dir="./output/photomaker/customization_outputs/mini-VGGFace2/"
export map_json_path="./customization/target_model/PhotoMaker/VGGFace2_max_photomaker_clip_distance.json"
export prompts="a_photo_of_sks_person;a_dslr_portrait_of_sks_person"
export VGGFace2="./datasets/VGGFace2"
echo $prompts

export save_config_dir="./output/photomaker/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/attack_gen_eval.sh" $save_config_dir

# python ./attack/faceoff.py \
#     --device $device \
#     --prior_generation_precision "bf16" \
#     --loss_type $loss_type \
#     --attack_num $attack_num \
#     --alpha $alpha \
#     --eps $eps \
#     --input_size $input_size \
#     --model_input_size $model_input_size \
#     --center_crop 1 \
#     --resample_interpolation 'BILINEAR' \
#     --data_dir "./datasets/${data_dir_name}" \
#     --input_name "set_B" \
#     --data_dir_for_target_max "./datasets/VGGFace2" \
#     --save_dir "./output/photomaker/adversarial_images" \
#     --model_type $model_type \
#     --pretrained_model_name_or_path "./pretrains/photomaker-v1.bin" \
#     --target_type $target_type \
#     --max_distance_json "./customization/target_model/PhotoMaker/VGGFace2_max_photomaker_clip_distance.json" \
#     --min_eps $min_eps \
#     --update_interval $update_interval \
#     --noise_budget_refiner $noise_budget_refiner \
#     --mean_filter_size $mean_filter_size \
#     --strong_edge $strong_edge \
#     --weak_edge $weak_edge

python ./customization/target_model/PhotoMaker/inference.py \
  --input_folders "./output/photomaker/adversarial_images/${adversarial_folder_name}" \
  --save_dir "./output/photomaker/customization_outputs/${adversarial_folder_name}" \
  --prompts "a photo of sks person;a dslr portrait of sks person" \
  --photomaker_ckpt "./pretrains/photomaker-v1.bin" \
  --base_model_path "./pretrains/RealVisXL_V3.0" \
  --device $device \
  --seed 42 \
  --num_steps 50 \
  --style_strength_ratio 20 \
  --num_images_per_prompt 4 \
  --pre_test 0 \
  --height 1024 \
  --width 1024 \
  --lora 0 \
  --input_name "" \
  --trigger_word "sks" \
  --gaussian_filter 0 \
  --hflip 0

# 1. IMS: protected output and original output
# ArcFace
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $customization_output_dir \
    --emb_dirs $original_output_dir \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --scene2 "original_output" \
    --is_target 0 \
    --map_path $map_json_path \
    --target_path "" \
    --model_name "ArcFace" \
    --input_name "" \
    --out_out 1
# VGG-Face
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $customization_output_dir \
    --emb_dirs $original_output_dir \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --scene2 "original_output" \
    --is_target 0 \
    --map_path $map_json_path \
    --target_path "" \
    --model_name "VGG-Face" \
    --input_name "" \
    --out_out 1
# CLIP
python ./evaluations/my_clip/my_clip.py \
    --prompts "a photo of sks person;a dslr portrait of sks person" \
    --data_dir $customization_output_dir \
    --emb_dirs $original_output_dir \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --scene2 "original_output" \
    --is_target 0 \
    --map_path $map_json_path \
    --target_path "" \
    --model_name_or_path "ViT-B/32" \
    --device $device \
    --input_name "" \
    --out_out 1

# 2. original_output and original_output
# ArcFace: default value is not 1, for failing to extract faces from some images
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $original_output_dir \
    --emb_dirs $original_output_dir \
    --save_dir $evaluation_output_dir \
    --scene "original_output" \
    --scene2 "original_output" \
    --is_target 0 \
    --map_path $map_json_path \
    --target_path "" \
    --model_name "ArcFace" \
    --input_name "" \
    --out_out 1
# VGG-Face: default value is not 1, for failing to extract faces from some images
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $original_output_dir \
    --emb_dirs $original_output_dir \
    --save_dir $evaluation_output_dir \
    --scene "original_output" \
    --scene2 "original_output" \
    --is_target 0 \
    --map_path $map_json_path \
    --target_path "" \
    --model_name "VGG-Face" \
    --input_name "" \
    --out_out 1
# CLIP: cosine similarity is 1 by default
python ./evaluations/my_clip/my_clip.py \
    --prompts $prompts \
    --data_dir $original_output_dir \
    --emb_dirs $original_output_dir \
    --save_dir $evaluation_output_dir \
    --scene "original_output" \
    --scene2 "original_output" \
    --is_target 0 \
    --map_path $map_json_path \
    --target_path "" \
    --model_name_or_path "ViT-B/32" \
    --device $device \
    --input_name "" \
    --out_out 1

# 3. OIMS: protected output and original input
# ArcFace
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $customization_output_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --scene2 "original_input" \
    --is_target 0 \
    --map_path $map_json_path \
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
    --map_path $map_json_path \
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
    --map_path $map_json_path \
    --target_path "" \
    --model_name_or_path "ViT-B/32" \
    --device $device \
    --input_name "" \
    --out_out 0

# 4. original output and original input
# ArcFace
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $original_output_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "original_output" \
    --scene2 "original_input" \
    --is_target 0 \
    --map_path $map_json_path \
    --target_path "" \
    --model_name "ArcFace" \
    --input_name "set_B" \
    --out_out 0
# VGG-Face
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $original_output_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "original_output" \
    --scene2 "original_input" \
    --is_target 0 \
    --map_path $map_json_path \
    --target_path "" \
    --model_name "VGG-Face" \
    --input_name "set_B" \
    --out_out 0
# CLIP
python ./evaluations/my_clip/my_clip.py \
    --prompts $prompts \
    --data_dir $original_output_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "original_output" \
    --scene2 "original_input" \
    --is_target 0 \
    --map_path $map_json_path \
    --target_path "" \
    --model_name_or_path "ViT-B/32" \
    --device $device \
    --input_name "" \
    --out_out 0


# 5. TIMS: protected output and target input
# ArcFace
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $customization_output_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --scene2 "target_input" \
    --is_target 1 \
    --map_path $map_json_path \
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
    --scene2 "target_input" \
    --is_target 1 \
    --map_path $map_json_path \
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
    --scene2 "target_input" \
    --is_target 1 \
    --map_path $map_json_path \
    --target_path "" \
    --model_name_or_path "ViT-B/32" \
    --device $device \
    --input_name "" \
    --out_out 0

# 6. original output and target input
# ArcFace
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $original_output_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "original_output" \
    --scene2 "target_input" \
    --is_target 1 \
    --map_path $map_json_path \
    --target_path "" \
    --model_name "ArcFace" \
    --input_name "set_B" \
    --out_out 0
# VGG-Face
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $original_output_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "original_output" \
    --scene2 "target_input" \
    --is_target 1 \
    --map_path $map_json_path \
    --target_path "" \
    --model_name "VGG-Face" \
    --input_name "set_B" \
    --out_out 0
# CLIP
python ./evaluations/my_clip/my_clip.py \
    --prompts $prompts \
    --data_dir $original_output_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "original_output" \
    --scene2 "target_input" \
    --is_target 1 \
    --map_path $map_json_path \
    --target_path "" \
    --model_name_or_path "ViT-B/32" \
    --device $device \
    --input_name "" \
    --out_out 0

# 7. image quality evaluation
# protected output
python ./evaluations/LIQE/run_liqe.py \
    --prompts $prompts \
    --data_dir $customization_output_dir \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --device $device
# resize input images to 224x224
# python ./evaluations/resize_to_224_image.py \
#     --data_dir ${adversarial_input_dir} \
#     --save_dir "${adversarial_input_dir}_224" \
#     --sub_folder "" \
#     --resolution 224
# protected input
python ./evaluations/LIQE/run_liqe_for_input.py \
    --data_dir $adversarial_input_dir \
    --sub_folder "" \
    --save_dir $evaluation_output_dir \
    --scene "protected_input" \
    --device $device
# lpips: protected_input and original_input
python ./evaluations/lpips/my_lpips.py \
    --data_dir $adversarial_input_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "protected_input" \
    --scene2 "original_input" \
    --model_name_or_path "alex" \
    --device $device \
    --resolution $input_size
