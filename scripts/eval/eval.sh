# cd evaluations
# cd deepface
# pip install -e .
# cd ..
# cd retinaface
# pip install -e .
# cd ../..

export dir_name="photomaker_clip_pre_test_x_num200_alpha6_eps16_input224_output224_max_refiner0"
export adversarial_input_dir="./output/adversarial_images/${dir_name}"
export customization_output_dir="./output/customization_outputs/photomaker/${dir_name}"
export evaluation_output_dir="./output/evaluation_outputs/${dir_name}"
export original_output_dir="./output/customization_outputs/photomaker/mini-VGGFace2/"
export map_json_path="./customization/target_model/PhotoMaker/VGGFace2_max_photomaker_clip_distance.json"
export device="cuda:2"
export prompts="a_photo_of_sks_person;a_dslr_portrait_of_sks_person"
export VGGFace2="./datasets/VGGFace2"
echo $prompts

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
    --data_dir ${adversarial_input_dir} \
    --sub_folder "" \
    --save_dir $evaluation_output_dir \
    --scene "protected_input" \
    --device $device
# lpips: protected_input and original_input
python ./evaluations/lpips/my_lpips.py \
    --data_dir ${adversarial_input_dir} \
    --emb_dirs ${VGGFace2} \
    --save_dir $evaluation_output_dir \
    --scene "protected_input" \
    --scene2 "original_input" \
    --model_name_or_path "alex" \
    --device $device \
    --resolution 224
