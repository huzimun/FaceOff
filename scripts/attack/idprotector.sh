export model_type="face_diffuser,photomaker,ipadapter,ipadapterplus" # "photomaker,ipadapterplus" # target models: photomaker，vae，clip，ipadapterplus, face_diffuser
export pretrained_model_name_or_path="/data1/humw/Pretrains/clip-vit-large-patch14,/data1/humw/Pretrains/photomaker-v1.bin,/data1/humw/Pretrains/IP-Adapter/sdxl_models/image_encoder,/data1/humw/Pretrains/IP-Adapter/models/image_encoder" #"/data1/humw/Pretrains/photomaker-v1.bin,/data1/humw/Pretrains/IP-Adapter/models/image_encoder"  # "/data1/humw/Pretrains/photomaker-v1.bin"，"/data1/humw/Pretrains/IP-Adapter/models/image_encoder"，"/data1/humw/Pretrains/stable-diffusion-2-1-base"
export data_dir_name="min-VGGFace2_224" #validation dataset
export w=1.0 # w=0.0, Ltgt; w=1.0, Ldevite; (1-w) * Ltgt + w * Ldevite
export attack_num=100 # attack iterations
export eps=9 # max noise budget
export alpha=1 # step size
export input_size=224 # original image size, protected image size
export target_type="none" # none for non-targeted, type of target image: max is max clip embedding mse distance, yingbu is beijing opera mask, mist is mist image, gray is gray image
export use_l1=0 # 1 for using l1, 0 for not using l1
export affine=1 # 1 for using affine, 0 for not using affine
export device="cuda:1"
export gau_kernel_size=7
export mode="idprotector"
export adversarial_folder_name="${model_type}_${data_dir_name}_w${w}_num${attack_num}_alpha${alpha}_eps${eps}_input${input_size}_gau${gau_kernel_size}_target-${target_type}_l1-${use_l1}_affine-${affine}_mode-${mode}"
echo $adversarial_folder_name
export adversarial_input_dir="./outputs/adversarial_images/${adversarial_folder_name}"

export customization_output_dir="./outputs/customization_outputs/${adversarial_folder_name}"
export evaluation_output_dir="./outputs/evaluation_outputs/${adversarial_folder_name}"
export map_json_path="/data1/humw/Codes/FaceOff/max_clip_cosine_distance_map.json"
export VGGFace2="./datasets/VGGFace2"

export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/attack/idprotector.sh" $save_config_dir

python3 ./attack/idprotector.py \
    --adversarial_folder_name ${adversarial_folder_name} \
    --mode $mode \
    --device $device \
    --prior_generation_precision "bf16" \
    --use_l1 $use_l1 \
    --affine $affine \
    --w $w \
    --attack_num $attack_num \
    --alpha $alpha \
    --eps $eps \
    --input_size $input_size \
    --gau_kernel_size $gau_kernel_size \
    --resample_interpolation "BILINEAR" \
    --data_dir "./datasets/${data_dir_name}" \
    --input_name "set_B" \
    --data_dir_for_target_max $VGGFace2 \
    --save_dir "./outputs/adversarial_images" \
    --model_type $model_type \
    --pretrained_model_name_or_path $pretrained_model_name_or_path \
    --target_type $target_type \
    --max_distance_json $map_json_path
