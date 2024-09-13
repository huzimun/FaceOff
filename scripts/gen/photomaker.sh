export adversarial_folder_name="photomaker_clip_pre_test_x_num200_alpha6_eps16_input224_output224_max_refiner0"
python ./customization/target_model/PhotoMaker/inference.py \
  --input_folders "./output/adversarial_images/${adversarial_folder_name}" \
  --save_dir "./output/customization_outputs/photomaker/${adversarial_folder_name}" \
  --prompts "a photo of sks person;a dslr portrait of sks person" \
  --photomaker_ckpt "./pretrains/photomaker-v1.bin" \
  --base_model_path "./pretrains/RealVisXL_V3.0" \
  --device "cuda:1" \
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


# python ./customization/target_model/PhotoMaker/inference.py \
#   --input_folders "/home/humw/Codes/FaceOff/datasets/mini-VGGFace2" \
#   --save_dir "./output/customization_outputs/photomaker/mini-VGGFace2" \
#   --prompts "a photo of sks person;a dslr portrait of sks person" \
#   --photomaker_ckpt "./pretrains/photomaker-v1.bin" \
#   --base_model_path "./pretrains/RealVisXL_V3.0" \
#   --device "cuda:1" \
#   --seed 42 \
#   --num_steps 50 \
#   --style_strength_ratio 20 \
#   --num_images_per_prompt 4 \
#   --pre_test 1 \
#   --height 1024 \
#   --width 1024 \
#   --lora 0 \
#   --input_name "set_B" \
#   --trigger_word "sks" \
#   --gaussian_filter 0 \
#   --hflip 0
