export adversarial_folder_name="ViT-B32,ViT-B16,ViT-L14_test_cosine_w0_num100_alpha0.005_eps16_input512_yingbu"
export device="cuda:7"
export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_photomaker.sh" $save_config_dir
python ./customization/target_model/PhotoMaker_v2/inference.py \
  --input_folders "./outputs/adversarial_images/${adversarial_folder_name}" \
  --save_dir "./outputs/customization_outputs/${adversarial_folder_name}" \
  --prompts "a photo of sks person;a dslr portrait of sks person" \
  --photomaker_ckpt "/home/humw/Pretrains/TencentARC/PhotoMaker-V2/photomaker-v2.bin" \
  --base_model_path "/home/humw/Pretrains/stabilityai/stable-diffusion-xl-base-1.0" \
  --device $device \
  --seed 42 \
  --num_steps 50 \
  --style_strength_ratio 20 \
  --num_images_per_prompt 16 \
  --pre_test 0 \
  --height 1024 \
  --width 1024 \
  --lora 0 \
  --input_name "" \
  --trigger_word "sks" \
  --gaussian_filter 0 \
  --hflip 0

# export adversarial_folder_name="VGGFace2"
# export device="cuda:7"
# export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
# mkdir $save_config_dir
# cp "./scripts/gen/gen_photomaker.sh" $save_config_dir
# python ./customization/target_model/PhotoMaker/inference.py \
#   --input_folders "./outputs/adversarial_images/${adversarial_folder_name}" \
#   --save_dir "./outputs/customization_outputs/${adversarial_folder_name}" \
#   --prompts "a photo of sks person;a dslr portrait of sks person" \
#   --photomaker_ckpt "./pretrains/photomaker-v1.bin" \
#   --base_model_path "./pretrains/stable-diffusion-xl-base-1.0" \
#   --device $device \
#   --seed 42 \
#   --num_steps 50 \
#   --style_strength_ratio 20 \
#   --num_images_per_prompt 16 \
#   --pre_test 0 \
#   --height 1024 \
#   --width 1024 \
#   --lora 0 \
#   --input_name "set_B" \
#   --trigger_word "sks" \
#   --gaussian_filter 0 \
#   --hflip 0
