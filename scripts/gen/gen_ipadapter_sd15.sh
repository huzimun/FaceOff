export adversarial_folder_name="face_diffuser,photomaker,ipadapter,ipadapterplus_min-VGGFace2_224_w1.0_num100_alpha1_eps9_input224_gau7_target-none_lpips-1_mode-no-projected"
export experiment_name="ip_adapter_sd15_${adversarial_folder_name}"
export device="cuda:1"

export save_config_dir="./outputs/config_scripts_logs/${experiment_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_ipadapter_sd15.sh" $save_config_dir

python3 ./customization/target_model/IP-Adapter/a_ip_adapter_sdxl_demo.py \
    --model_type "sd15" \
    --base_model_path "/data1/humw/Pretrains/stable-diffusion-v1-5" \
    --image_encoder_path "/data1/humw/Pretrains/IP-Adapter/models/image_encoder" \
    --ip_ckpt "/data1/humw/Pretrains/IP-Adapter/models/ip-adapter_sd15.bin" \
    --vae_model_path "/data1/humw/Pretrains/sd-vae-ft-mse" \
    --device $device \
    --input_dir "./outputs/adversarial_images/${adversarial_folder_name}" \
    --output_dir "./outputs/customization_outputs/${experiment_name}" \
    --resolution 224 \
    --sub_name "" \
    --prior_generation_precision "fp16"
