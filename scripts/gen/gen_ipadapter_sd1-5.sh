export adversarial_folder_name="Encoder_attack_conda-photomaker_VGGFace2_vae15-ipadapter-photomaker_cosine_eot-0_min-mask_agm-0_norm-0"
export experiment_name="Encoder_attack_conda-photomaker_VGGFace2_vae15-ipadapter-photomaker_cosine_eot-0_min-mask_agm-0_norm-0"
export device="cuda:3"
export save_config_dir="./outputs/config_scripts_logs/${experiment_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_ipadapter_sd1-5.sh" $save_config_dir
python3 ./customization/target_model/IP-Adapter/a_ip_adapter_sdxl_plus-face_demo.py \
    --model_type "sd15" \
    --base_model_path "/data1/humw/Pretrains/stable-diffusion-v1-5" \
    --image_encoder_path "/data1/humw/Pretrains/IP-Adapter/models/image_encoder" \
    --ip_ckpt "/data1/humw/Pretrains/IP-Adapter/models/ip-adapter-plus_sd15.bin" \
    --vae_model_path "/data1/humw/Pretrains/sd-vae-ft-mse" \
    --device $device \
    --input_dir "./outputs/adversarial_images/${adversarial_folder_name}" \
    --output_dir "./outputs/customization_outputs/${experiment_name}" \
    --resolution 224 \
    --sub_name "" \
    --prior_generation_precision "fp16"

export adversarial_folder_name="Encoder_attack_conda-photomaker_VGGFace2_vae15-ipadapter-photomaker_cosine_eot-0_max-mask_agm-0_norm-0"
export experiment_name="Encoder_attack_conda-photomaker_VGGFace2_vae15-ipadapter-photomaker_cosine_eot-0_max-mask_agm-0_norm-0"
export device="cuda:3"
export save_config_dir="./outputs/config_scripts_logs/${experiment_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_ipadapter_sd1-5.sh" $save_config_dir
python3 ./customization/target_model/IP-Adapter/a_ip_adapter_sdxl_plus-face_demo.py \
    --model_type "sd15" \
    --base_model_path "/data1/humw/Pretrains/stable-diffusion-v1-5" \
    --image_encoder_path "/data1/humw/Pretrains/IP-Adapter/models/image_encoder" \
    --ip_ckpt "/data1/humw/Pretrains/IP-Adapter/models/ip-adapter-plus_sd15.bin" \
    --vae_model_path "/data1/humw/Pretrains/sd-vae-ft-mse" \
    --device $device \
    --input_dir "./outputs/adversarial_images/${adversarial_folder_name}" \
    --output_dir "./outputs/customization_outputs/${experiment_name}" \
    --resolution 224 \
    --sub_name "" \
    --prior_generation_precision "fp16"

export adversarial_folder_name="Encoder_attack_conda-photomaker_VGGFace2_vae15-ipadapter-photomaker_cosine_eot-0_random-mask_agm-0_norm-0"
export experiment_name="Encoder_attack_conda-photomaker_VGGFace2_vae15-ipadapter-photomaker_cosine_eot-0_random-mask_agm-0_norm-0"
export device="cuda:3"
export save_config_dir="./outputs/config_scripts_logs/${experiment_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_ipadapter_sd1-5.sh" $save_config_dir
python3 ./customization/target_model/IP-Adapter/a_ip_adapter_sdxl_plus-face_demo.py \
    --model_type "sd15" \
    --base_model_path "/data1/humw/Pretrains/stable-diffusion-v1-5" \
    --image_encoder_path "/data1/humw/Pretrains/IP-Adapter/models/image_encoder" \
    --ip_ckpt "/data1/humw/Pretrains/IP-Adapter/models/ip-adapter-plus_sd15.bin" \
    --vae_model_path "/data1/humw/Pretrains/sd-vae-ft-mse" \
    --device $device \
    --input_dir "./outputs/adversarial_images/${adversarial_folder_name}" \
    --output_dir "./outputs/customization_outputs/${experiment_name}" \
    --resolution 224 \
    --sub_name "" \
    --prior_generation_precision "fp16"
