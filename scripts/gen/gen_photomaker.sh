export adversarial_folder_name="Encoder_attack_conda-photomaker_8-255_VGGFace2_vae15-ipadapter-photomaker_mix_eot-0_yingbu_agm-2_norm-0"
export device="cuda:0"

export BASE_MODEL="SDXL-BASE-1"  # "SDXL-BASE-1" RealVisXL_V3 RealVisXL_V4
if [ "$BASE_MODEL" = "SDXL-BASE-1" ]; then
  export base_model_path="/data1/humw/Pretrains/stable-diffusion-xl-base-1.0"
elif [ "$BASE_MODEL" = "RealVisXL_V3" ]; then
  export base_model_path="/data1/humw/Pretrains/RealVisXL-V3.0"
elif [ "$BASE_MODEL" = "RealVisXL_V4" ]; then
  export base_model_path="/data1/humw/Pretrains/SG161222/RealVisXL_V4.0"
else
  echo "Invalid BASE_MODEL"
  exit 1
fi

export lora=0

export outputs_folder_name="photomaker_${BASE_MODEL}_${adversarial_folder_name}_lora-${lora}"

export save_config_dir="./outputs/config_scripts_logs/${outputs_folder_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_photomaker.sh" $save_config_dir

python3 ./customization/target_model/PhotoMaker/inference.py \
  --input_folders "./outputs/adversarial_images/${adversarial_folder_name}" \
  --save_dir "./outputs/customization_outputs/${outputs_folder_name}" \
  --prompts "a photo of sks person;a dslr portrait of sks person" \
  --photomaker_ckpt "/data1/humw/Pretrains/photomaker-v1.bin" \
  --base_model_path ${base_model_path} \
  --device $device \
  --seed 42 \
  --num_steps 50 \
  --style_strength_ratio 20 \
  --num_images_per_prompt 16 \
  --pre_test 0 \
  --height 1024 \
  --width 1024 \
  --lora $lora \
  --input_name "" \
  --trigger_word "sks" \
  --gaussian_filter 0 \
  --hflip 0


export adversarial_folder_name="Encoder_attack_conda-photomaker_24-255_VGGFace2_vae15-ipadapter-photomaker_mix_eot-0_yingbu_agm-2_norm-0"
export device="cuda:0"

export BASE_MODEL="SDXL-BASE-1"  # "SDXL-BASE-1" RealVisXL_V3 RealVisXL_V4
if [ "$BASE_MODEL" = "SDXL-BASE-1" ]; then
  export base_model_path="/data1/humw/Pretrains/stable-diffusion-xl-base-1.0"
elif [ "$BASE_MODEL" = "RealVisXL_V3" ]; then
  export base_model_path="/data1/humw/Pretrains/RealVisXL-V3.0"
elif [ "$BASE_MODEL" = "RealVisXL_V4" ]; then
  export base_model_path="/data1/humw/Pretrains/SG161222/RealVisXL_V4.0"
else
  echo "Invalid BASE_MODEL"
  exit 1
fi

export lora=0

export outputs_folder_name="photomaker_${BASE_MODEL}_${adversarial_folder_name}_lora-${lora}"

export save_config_dir="./outputs/config_scripts_logs/${outputs_folder_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_photomaker.sh" $save_config_dir

python3 ./customization/target_model/PhotoMaker/inference.py \
  --input_folders "./outputs/adversarial_images/${adversarial_folder_name}" \
  --save_dir "./outputs/customization_outputs/${outputs_folder_name}" \
  --prompts "a photo of sks person;a dslr portrait of sks person" \
  --photomaker_ckpt "/data1/humw/Pretrains/photomaker-v1.bin" \
  --base_model_path ${base_model_path} \
  --device $device \
  --seed 42 \
  --num_steps 50 \
  --style_strength_ratio 20 \
  --num_images_per_prompt 16 \
  --pre_test 0 \
  --height 1024 \
  --width 1024 \
  --lora $lora \
  --input_name "" \
  --trigger_word "sks" \
  --gaussian_filter 0 \
  --hflip 0



export adversarial_folder_name="Encoder_attack_conda-photomaker_32-255_VGGFace2_vae15-ipadapter-photomaker_mix_eot-0_yingbu_agm-2_norm-0"
export device="cuda:0"

export BASE_MODEL="SDXL-BASE-1"  # "SDXL-BASE-1" RealVisXL_V3 RealVisXL_V4
if [ "$BASE_MODEL" = "SDXL-BASE-1" ]; then
  export base_model_path="/data1/humw/Pretrains/stable-diffusion-xl-base-1.0"
elif [ "$BASE_MODEL" = "RealVisXL_V3" ]; then
  export base_model_path="/data1/humw/Pretrains/RealVisXL-V3.0"
elif [ "$BASE_MODEL" = "RealVisXL_V4" ]; then
  export base_model_path="/data1/humw/Pretrains/SG161222/RealVisXL_V4.0"
else
  echo "Invalid BASE_MODEL"
  exit 1
fi

export lora=0

export outputs_folder_name="photomaker_${BASE_MODEL}_${adversarial_folder_name}_lora-${lora}"

export save_config_dir="./outputs/config_scripts_logs/${outputs_folder_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_photomaker.sh" $save_config_dir

python3 ./customization/target_model/PhotoMaker/inference.py \
  --input_folders "./outputs/adversarial_images/${adversarial_folder_name}" \
  --save_dir "./outputs/customization_outputs/${outputs_folder_name}" \
  --prompts "a photo of sks person;a dslr portrait of sks person" \
  --photomaker_ckpt "/data1/humw/Pretrains/photomaker-v1.bin" \
  --base_model_path ${base_model_path} \
  --device $device \
  --seed 42 \
  --num_steps 50 \
  --style_strength_ratio 20 \
  --num_images_per_prompt 16 \
  --pre_test 0 \
  --height 1024 \
  --width 1024 \
  --lora $lora \
  --input_name "" \
  --trigger_word "sks" \
  --gaussian_filter 0 \
  --hflip 0



export adversarial_folder_name="Encoder_attack_conda-photomaker_VGGFace2_vae15-ipadapter-photomaker_mix_eot-0_baozheng_agm-2_norm-0"
export device="cuda:0"

export BASE_MODEL="SDXL-BASE-1"  # "SDXL-BASE-1" RealVisXL_V3 RealVisXL_V4
if [ "$BASE_MODEL" = "SDXL-BASE-1" ]; then
  export base_model_path="/data1/humw/Pretrains/stable-diffusion-xl-base-1.0"
elif [ "$BASE_MODEL" = "RealVisXL_V3" ]; then
  export base_model_path="/data1/humw/Pretrains/RealVisXL-V3.0"
elif [ "$BASE_MODEL" = "RealVisXL_V4" ]; then
  export base_model_path="/data1/humw/Pretrains/SG161222/RealVisXL_V4.0"
else
  echo "Invalid BASE_MODEL"
  exit 1
fi

export lora=0

export outputs_folder_name="photomaker_${BASE_MODEL}_${adversarial_folder_name}_lora-${lora}"

export save_config_dir="./outputs/config_scripts_logs/${outputs_folder_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_photomaker.sh" $save_config_dir

python3 ./customization/target_model/PhotoMaker/inference.py \
  --input_folders "./outputs/adversarial_images/${adversarial_folder_name}" \
  --save_dir "./outputs/customization_outputs/${outputs_folder_name}" \
  --prompts "a photo of sks person;a dslr portrait of sks person" \
  --photomaker_ckpt "/data1/humw/Pretrains/photomaker-v1.bin" \
  --base_model_path ${base_model_path} \
  --device $device \
  --seed 42 \
  --num_steps 50 \
  --style_strength_ratio 20 \
  --num_images_per_prompt 16 \
  --pre_test 0 \
  --height 1024 \
  --width 1024 \
  --lora $lora \
  --input_name "" \
  --trigger_word "sks" \
  --gaussian_filter 0 \
  --hflip 0



export adversarial_folder_name="Encoder_attack_conda-photomaker_VGGFace2_vae15-ipadapter-photomaker_mix_eot-0_caocao_agm-2_norm-0"
export device="cuda:0"

export BASE_MODEL="SDXL-BASE-1"  # "SDXL-BASE-1" RealVisXL_V3 RealVisXL_V4
if [ "$BASE_MODEL" = "SDXL-BASE-1" ]; then
  export base_model_path="/data1/humw/Pretrains/stable-diffusion-xl-base-1.0"
elif [ "$BASE_MODEL" = "RealVisXL_V3" ]; then
  export base_model_path="/data1/humw/Pretrains/RealVisXL-V3.0"
elif [ "$BASE_MODEL" = "RealVisXL_V4" ]; then
  export base_model_path="/data1/humw/Pretrains/SG161222/RealVisXL_V4.0"
else
  echo "Invalid BASE_MODEL"
  exit 1
fi

export lora=0

export outputs_folder_name="photomaker_${BASE_MODEL}_${adversarial_folder_name}_lora-${lora}"

export save_config_dir="./outputs/config_scripts_logs/${outputs_folder_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_photomaker.sh" $save_config_dir

python3 ./customization/target_model/PhotoMaker/inference.py \
  --input_folders "./outputs/adversarial_images/${adversarial_folder_name}" \
  --save_dir "./outputs/customization_outputs/${outputs_folder_name}" \
  --prompts "a photo of sks person;a dslr portrait of sks person" \
  --photomaker_ckpt "/data1/humw/Pretrains/photomaker-v1.bin" \
  --base_model_path ${base_model_path} \
  --device $device \
  --seed 42 \
  --num_steps 50 \
  --style_strength_ratio 20 \
  --num_images_per_prompt 16 \
  --pre_test 0 \
  --height 1024 \
  --width 1024 \
  --lora $lora \
  --input_name "" \
  --trigger_word "sks" \
  --gaussian_filter 0 \
  --hflip 0




export adversarial_folder_name="Encoder_attack_conda-photomaker_VGGFace2_vae15-ipadapter-photomaker_mix_eot-0_chengyaojin_agm-2_norm-0"
export device="cuda:0"

export BASE_MODEL="SDXL-BASE-1"  # "SDXL-BASE-1" RealVisXL_V3 RealVisXL_V4
if [ "$BASE_MODEL" = "SDXL-BASE-1" ]; then
  export base_model_path="/data1/humw/Pretrains/stable-diffusion-xl-base-1.0"
elif [ "$BASE_MODEL" = "RealVisXL_V3" ]; then
  export base_model_path="/data1/humw/Pretrains/RealVisXL-V3.0"
elif [ "$BASE_MODEL" = "RealVisXL_V4" ]; then
  export base_model_path="/data1/humw/Pretrains/SG161222/RealVisXL_V4.0"
else
  echo "Invalid BASE_MODEL"
  exit 1
fi

export lora=0

export outputs_folder_name="photomaker_${BASE_MODEL}_${adversarial_folder_name}_lora-${lora}"

export save_config_dir="./outputs/config_scripts_logs/${outputs_folder_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_photomaker.sh" $save_config_dir

python3 ./customization/target_model/PhotoMaker/inference.py \
  --input_folders "./outputs/adversarial_images/${adversarial_folder_name}" \
  --save_dir "./outputs/customization_outputs/${outputs_folder_name}" \
  --prompts "a photo of sks person;a dslr portrait of sks person" \
  --photomaker_ckpt "/data1/humw/Pretrains/photomaker-v1.bin" \
  --base_model_path ${base_model_path} \
  --device $device \
  --seed 42 \
  --num_steps 50 \
  --style_strength_ratio 20 \
  --num_images_per_prompt 16 \
  --pre_test 0 \
  --height 1024 \
  --width 1024 \
  --lora $lora \
  --input_name "" \
  --trigger_word "sks" \
  --gaussian_filter 0 \
  --hflip 0




export adversarial_folder_name="Encoder_attack_conda-photomaker_VGGFace2_vae15-ipadapter-photomaker_mix_eot-0_guanyu_agm-2_norm-0"
export device="cuda:0"

export BASE_MODEL="SDXL-BASE-1"  # "SDXL-BASE-1" RealVisXL_V3 RealVisXL_V4
if [ "$BASE_MODEL" = "SDXL-BASE-1" ]; then
  export base_model_path="/data1/humw/Pretrains/stable-diffusion-xl-base-1.0"
elif [ "$BASE_MODEL" = "RealVisXL_V3" ]; then
  export base_model_path="/data1/humw/Pretrains/RealVisXL-V3.0"
elif [ "$BASE_MODEL" = "RealVisXL_V4" ]; then
  export base_model_path="/data1/humw/Pretrains/SG161222/RealVisXL_V4.0"
else
  echo "Invalid BASE_MODEL"
  exit 1
fi

export lora=0

export outputs_folder_name="photomaker_${BASE_MODEL}_${adversarial_folder_name}_lora-${lora}"

export save_config_dir="./outputs/config_scripts_logs/${outputs_folder_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_photomaker.sh" $save_config_dir

python3 ./customization/target_model/PhotoMaker/inference.py \
  --input_folders "./outputs/adversarial_images/${adversarial_folder_name}" \
  --save_dir "./outputs/customization_outputs/${outputs_folder_name}" \
  --prompts "a photo of sks person;a dslr portrait of sks person" \
  --photomaker_ckpt "/data1/humw/Pretrains/photomaker-v1.bin" \
  --base_model_path ${base_model_path} \
  --device $device \
  --seed 42 \
  --num_steps 50 \
  --style_strength_ratio 20 \
  --num_images_per_prompt 16 \
  --pre_test 0 \
  --height 1024 \
  --width 1024 \
  --lora $lora \
  --input_name "" \
  --trigger_word "sks" \
  --gaussian_filter 0 \
  --hflip 0




export adversarial_folder_name="Encoder_attack_conda-photomaker_VGGFace2_vae15-ipadapter-photomaker_mix_eot-0_lumeng_agm-2_norm-0"
export device="cuda:0"

export BASE_MODEL="SDXL-BASE-1"  # "SDXL-BASE-1" RealVisXL_V3 RealVisXL_V4
if [ "$BASE_MODEL" = "SDXL-BASE-1" ]; then
  export base_model_path="/data1/humw/Pretrains/stable-diffusion-xl-base-1.0"
elif [ "$BASE_MODEL" = "RealVisXL_V3" ]; then
  export base_model_path="/data1/humw/Pretrains/RealVisXL-V3.0"
elif [ "$BASE_MODEL" = "RealVisXL_V4" ]; then
  export base_model_path="/data1/humw/Pretrains/SG161222/RealVisXL_V4.0"
else
  echo "Invalid BASE_MODEL"
  exit 1
fi

export lora=0

export outputs_folder_name="photomaker_${BASE_MODEL}_${adversarial_folder_name}_lora-${lora}"

export save_config_dir="./outputs/config_scripts_logs/${outputs_folder_name}"
mkdir $save_config_dir
cp "./scripts/gen/gen_photomaker.sh" $save_config_dir

python3 ./customization/target_model/PhotoMaker/inference.py \
  --input_folders "./outputs/adversarial_images/${adversarial_folder_name}" \
  --save_dir "./outputs/customization_outputs/${outputs_folder_name}" \
  --prompts "a photo of sks person;a dslr portrait of sks person" \
  --photomaker_ckpt "/data1/humw/Pretrains/photomaker-v1.bin" \
  --base_model_path ${base_model_path} \
  --device $device \
  --seed 42 \
  --num_steps 50 \
  --style_strength_ratio 20 \
  --num_images_per_prompt 16 \
  --pre_test 0 \
  --height 1024 \
  --width 1024 \
  --lora $lora \
  --input_name "" \
  --trigger_word "sks" \
  --gaussian_filter 0 \
  --hflip 0
