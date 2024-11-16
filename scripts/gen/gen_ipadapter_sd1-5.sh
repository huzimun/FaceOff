# export adversarial_folder_name="VGGFace2"
# export experiment_name="VGGFace2_IP-Adapter"
# export device="cuda:0"
# export save_config_dir="./outputs/config_scripts_logs/${experiment_name}"
# mkdir $save_config_dir
# cp "./scripts/gen/gen_ipadapter_sd1-5.sh" $save_config_dir
# python ./customization/target_model/IP-Adapter/a_ip_adapter_sdxl_plus-face_demo.py \
#     --model_type "sd15" \
#     --base_model_path "/data1/humw/Pretrains/stable-diffusion-v1-5" \
#     --image_encoder_path "/data1/humw/Pretrains/IP-Adapter/models/image_encoder" \
#     --ip_ckpt "/data1/humw/Pretrains/IP-Adapter/models/ip-adapter-plus_sd15.bin" \
#     --vae_model_path "/data1/humw/Pretrains/sd-vae-ft-mse" \
#     --device $device \
#     --input_dir "./outputs/adversarial_images/${adversarial_folder_name}" \
#     --output_dir "./outputs/customization_outputs/${experiment_name}" \
#     --resolution 224 \
#     --sub_name "set_B" \
#     --prior_generation_precision "fp16"

export adversarial_folder_name="ipadapter_VGGFace2_w0.0_num50_alpha6_eps16_input512_224_yingbu_refiner0"
export experiment_name="ipadapter_VGGFace2_w0.0_num50_alpha6_eps16_input512_224_yingbu_refiner0"
export device="cuda:0"
export save_config_dir="./outputs/config_scripts_logs/${experiment_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_ipadapter_sd1-5.sh" $save_config_dir
python ./customization/target_model/IP-Adapter/a_ip_adapter_sdxl_plus-face_demo.py \
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
