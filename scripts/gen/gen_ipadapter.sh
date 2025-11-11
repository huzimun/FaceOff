export adversarial_folder_name="VGGFace2"
export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_ipadapter.sh" $save_config_dir
export device="cuda:5"

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
    --sub_name "set_B" \
    --prior_generation_precision "fp16"

export adversarial_folder_name="new-CelebA-HQ"
export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_ipadapter.sh" $save_config_dir
export device="cuda:5"

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
    --sub_name "set_B" \
    --prior_generation_precision "fp16"
