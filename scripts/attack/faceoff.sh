# 方式1
python ./attack/faceoff.py --config_path "./args.json"

# 方式2
# python ./attack/faceoff.py \
#     --device "cuda:0" \
#     --prior_generation_precision "bf16" \
#     --loss_type "n" \
#     --attack_num 200 \
#     --alpha 6 \
#     --eps 16 \
#     --input_size 224 \
#     --output_size 224 \
#     --center_crop 1 \
#     --resample_interpolation 'BILINEAR' \
#     --data_dir "./datasets/pre_test" \
#     --input_name "set_B" \
#     --data_dir_for_target_max "./datasets/VGGFace2" \
#     --save_dir "./output/photomaker" \
#     --model_type "photomaker_clip" \
#     --pretrained_model_name_or_path "./pretrains/photomaker-v1.bin" \
#     --target_type 'max' \
#     --max_distance_json "./customization/target_model/PhotoMaker/VGGFace2_max_photomaker_clip_distance.json" \
#     --min_JND_eps_rate 0.75 \
#     --update_interval 40 \
#     --noise_budget_refiner 0 \
#     --blur_size 3 \