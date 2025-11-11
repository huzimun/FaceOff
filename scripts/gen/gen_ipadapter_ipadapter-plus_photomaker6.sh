
export adversarial_folder_name="ipadapter,ipadapter-plus,photomaker_VGGFace2_cosine_w0.5_num100_alpha0.005_eps16_input512_yingbu_grdipure"
export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_ipadapter_ipadapter-plus_photomaker6.sh" $save_config_dir
export device="cuda:2"

export experiment_name="ipadapter_sdxl_"$adversarial_folder_name
python ./customization/target_model/IP-Adapter/a_ip_adapter_sdxl_demo.py \
    --model_type "sdxl" \
    --base_model_path "/home/humw/Pretrains/stabilityai/stable-diffusion-xl-base-1.0" \
    --image_encoder_path "/home/humw/Pretrains/h94/IP-Adapter/sdxl_models/image_encoder" \
    --ip_ckpt "/home/humw/Pretrains/h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin" \
    --vae_model_path "/home/humw/Pretrains/sd-vae-ft-mse" \
    --device $device \
    --input_dir "./outputs/adversarial_images/${adversarial_folder_name}" \
    --output_dir "./outputs/customization_outputs/${experiment_name}" \
    --resolution 224 \
    --sub_name "" \
    --prior_generation_precision "fp16"

export experiment_name="ipadapter-plus_sdxl_"$adversarial_folder_name
python ./customization/target_model/IP-Adapter/a_ip_adapter_sdxl_plus-face_demo.py \
    --model_type "sdxl" \
    --base_model_path "/home/humw/Pretrains/stabilityai/stable-diffusion-xl-base-1.0" \
    --image_encoder_path "/home/humw/Pretrains/h94/IP-Adapter/models/image_encoder" \
    --ip_ckpt "/home/humw/Pretrains/h94/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin" \
    --vae_model_path "/home/humw/Pretrains/sd-vae-ft-mse" \
    --device $device \
    --input_dir "./outputs/adversarial_images/${adversarial_folder_name}" \
    --output_dir "./outputs/customization_outputs/${experiment_name}" \
    --resolution 224 \
    --sub_name "" \
    --prior_generation_precision "fp16"

export experiment_name="photomaker_sdxl_"$adversarial_folder_name
python ./customization/target_model/PhotoMaker/inference.py \
  --input_folders "./outputs/adversarial_images/${adversarial_folder_name}" \
  --save_dir "./outputs/customization_outputs/${experiment_name}" \
  --prompts "a photo of sks person;a dslr portrait of sks person" \
  --photomaker_ckpt "/home/humw/Pretrains/photomaker-v1.bin" \
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
