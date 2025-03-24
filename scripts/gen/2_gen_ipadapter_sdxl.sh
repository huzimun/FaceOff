export adversarial_folder_name="IDProtector_conda-photomaker_VGGFace2_ipadapter_cosine_eot-0_non-target_agm-0_norm-0"
export experiment_name="IDProtector_conda-photomaker_VGGFace2_ipadapter_cosine_eot-0_non-target_agm-0_norm-0"
export device="cuda:0"
export save_config_dir="./outputs/config_scripts_logs/${experiment_name}"
mkdir $save_config_dir
cp "./scripts/gen/2_gen_ipadapter_sdxl.sh" $save_config_dir
python3 ./customization/target_model/IP-Adapter/a_ip_adapter_sdxl_plus-face_demo.py \
    --model_type "sdxl" \
    --base_model_path "/data1/humw/Pretrains/stable-diffusion-xl-base-1.0" \
    --image_encoder_path "/data1/humw/Pretrains/IP-Adapter/models/image_encoder" \
    --ip_ckpt "/data1/humw/Pretrains/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin" \
    --vae_model_path "/data1/humw/Pretrains/sd-vae-ft-mse" \
    --device $device \
    --input_dir "./outputs/adversarial_images/${adversarial_folder_name}" \
    --output_dir "./outputs/customization_outputs/${experiment_name}" \
    --resolution 224 \
    --sub_name "" \
    --prior_generation_precision "fp16"

export adversarial_folder_name="IDProtector_conda-photomaker_VGGFace2_ipadapter-photomaker_cosine_eot-0_non-target_agm-0_norm-0"
export experiment_name="IDProtector_conda-photomaker_VGGFace2_ipadapter-photomaker_cosine_eot-0_non-target_agm-0_norm-0"
export device="cuda:0"
export save_config_dir="./outputs/config_scripts_logs/${experiment_name}"
mkdir $save_config_dir
cp "./scripts/gen/2_gen_ipadapter_sdxl.sh" $save_config_dir
python3 ./customization/target_model/IP-Adapter/a_ip_adapter_sdxl_plus-face_demo.py \
    --model_type "sdxl" \
    --base_model_path "/data1/humw/Pretrains/stable-diffusion-xl-base-1.0" \
    --image_encoder_path "/data1/humw/Pretrains/IP-Adapter/models/image_encoder" \
    --ip_ckpt "/data1/humw/Pretrains/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin" \
    --vae_model_path "/data1/humw/Pretrains/sd-vae-ft-mse" \
    --device $device \
    --input_dir "./outputs/adversarial_images/${adversarial_folder_name}" \
    --output_dir "./outputs/customization_outputs/${experiment_name}" \
    --resolution 224 \
    --sub_name "" \
    --prior_generation_precision "fp16"
