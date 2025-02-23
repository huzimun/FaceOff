export adversarial_folder_name="photomaker_clip_VGGFace2_w0.0_num100_alpha6_eps16_input512_224_mist_refiner0"
export device="cuda:2"
export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_photomaker.sh" $save_config_dir
python ./customization/target_model/PhotoMaker/inference.py \
  --input_folders "./outputs/adversarial_images/${adversarial_folder_name}" \
  --save_dir "./outputs/customization_outputs/${adversarial_folder_name}" \
  --prompts "a photo of sks person;a dslr portrait of sks person" \
  --photomaker_ckpt "./pretrains/photomaker-v1.bin" \
  --base_model_path "./pretrains/stable-diffusion-xl-base-1.0" \
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

export adversarial_folder_name="VGGFace2"
export device="cuda:2"
export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_photomaker.sh" $save_config_dir
python ./customization/target_model/PhotoMaker/inference.py \
  --input_folders "./outputs/adversarial_images/${adversarial_folder_name}" \
  --save_dir "./outputs/customization_outputs/${adversarial_folder_name}" \
  --prompts "a photo of sks person;a dslr portrait of sks person" \
  --photomaker_ckpt "./pretrains/photomaker-v1.bin" \
  --base_model_path "./pretrains/stable-diffusion-xl-base-1.0" \
  --device $device \
  --seed 42 \
  --num_steps 50 \
  --style_strength_ratio 20 \
  --num_images_per_prompt 16 \
  --pre_test 0 \
  --height 1024 \
  --width 1024 \
  --lora 0 \
  --input_name "set_B" \
  --trigger_word "sks" \
  --gaussian_filter 0 \
  --hflip 0
