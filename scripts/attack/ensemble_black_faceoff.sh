export model_type="ViT-B32,ViT-B16,ViT-L14" # 攻击的模型，photomaker_clip，vae，clip，ipadapter
export pretrained_model_name_or_path="ViT-B/32,ViT-B/16,ViT-L/14"  # "/data1/humw/Pretrains/photomaker-v1.bin"，"/data1/humw/Pretrains/IP-Adapter/models/image_encoder"，"/data1/humw/Pretrains/stable-diffusion-2-1-base"
export data_dir_name="test" # 输入数据集
export w=0.25 # w=0.0, x; w=1.0, d; (1-w) * Ltgt + w * Ldevite
export attack_num=100 # 攻击轮次
export alpha=0.005 # 步长，与已有方法一致
export eps=16 # 最大噪声阈值
export input_size=512 # 输入图片的尺寸，对抗图片尺寸
export target_type="yingbu" # target图片的类型，max代表最大clip特征mse距离, yingbu代表target image为yingbu, mist代表mist图像
export device="cuda:5"
export loss_choice="cosine" # mse or cosine
echo $noise_budget_refiner

export adversarial_folder_name="${model_type}_${data_dir_name}_${loss_choice}_w${w}_num${attack_num}_alpha${alpha}_eps${eps}_input${input_size}_${target_type}";

echo $adversarial_folder_name
export adversarial_input_dir="./outputs/adversarial_images/${adversarial_folder_name}"
export customization_output_dir="./outputs/customization_outputs/${adversarial_folder_name}"
export evaluation_output_dir="./outputs/evaluation_outputs/${adversarial_folder_name}"
export map_json_path="./max_clip_cosine_distance_map.json"
# export prompts="a_photo_of_person;a_dslr_portrait_of_person" # "a_photo_of_sks_person;a_dslr_portrait_of_sks_person"
export VGGFace2="./datasets/VGGFace2"
# echo $prompts

export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/attack/ensemble_faceoff.sh" $save_config_dir

python ./attack/ensemble_faceoff.py \
    --adversarial_folder_name $adversarial_folder_name \
    --device $device \
    --prior_generation_precision "fp32" \
    --loss_choice $loss_choice \
    --w $w \
    --attack_num $attack_num \
    --alpha $alpha \
    --eps $eps \
    --input_size $input_size \
    --center_crop 1 \
    --resample_interpolation "BILINEAR" \
    --data_dir "./datasets/${data_dir_name}" \
    --input_name "set_B" \
    --data_dir_for_target_max $VGGFace2 \
    --save_dir "./outputs/adversarial_images" \
    --model_type $model_type \
    --pretrained_model_name_or_path $pretrained_model_name_or_path \
    --target_type $target_type \
    --max_distance_json $map_json_path

export model_type="ViT-B32,ViT-B16,ViT-L14" # 攻击的模型，photomaker_clip，vae，clip，ipadapter
export pretrained_model_name_or_path="ViT-B/32,ViT-B/16,ViT-L/14"  # "/data1/humw/Pretrains/photomaker-v1.bin"，"/data1/humw/Pretrains/IP-Adapter/models/image_encoder"，"/data1/humw/Pretrains/stable-diffusion-2-1-base"
export data_dir_name="test" # 输入数据集
export w=0.5 # w=0.0, x; w=1.0, d; (1-w) * Ltgt + w * Ldevite
export attack_num=100 # 攻击轮次
export alpha=0.005 # 步长，与已有方法一致
export eps=16 # 最大噪声阈值
export input_size=512 # 输入图片的尺寸，对抗图片尺寸
export target_type="yingbu" # target图片的类型，max代表最大clip特征mse距离, yingbu代表target image为yingbu, mist代表mist图像
export device="cuda:5"
export loss_choice="cosine" # mse or cosine
echo $noise_budget_refiner

export adversarial_folder_name="${model_type}_${data_dir_name}_${loss_choice}_w${w}_num${attack_num}_alpha${alpha}_eps${eps}_input${input_size}_${target_type}";

echo $adversarial_folder_name
export adversarial_input_dir="./outputs/adversarial_images/${adversarial_folder_name}"
export customization_output_dir="./outputs/customization_outputs/${adversarial_folder_name}"
export evaluation_output_dir="./outputs/evaluation_outputs/${adversarial_folder_name}"
export map_json_path="./max_clip_cosine_distance_map.json"
# export prompts="a_photo_of_person;a_dslr_portrait_of_person" # "a_photo_of_sks_person;a_dslr_portrait_of_sks_person"
export VGGFace2="./datasets/VGGFace2"
# echo $prompts

export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/attack/ensemble_faceoff.sh" $save_config_dir

python ./attack/ensemble_faceoff.py \
    --adversarial_folder_name $adversarial_folder_name \
    --device $device \
    --prior_generation_precision "fp32" \
    --loss_choice $loss_choice \
    --w $w \
    --attack_num $attack_num \
    --alpha $alpha \
    --eps $eps \
    --input_size $input_size \
    --center_crop 1 \
    --resample_interpolation "BILINEAR" \
    --data_dir "./datasets/${data_dir_name}" \
    --input_name "set_B" \
    --data_dir_for_target_max $VGGFace2 \
    --save_dir "./outputs/adversarial_images" \
    --model_type $model_type \
    --pretrained_model_name_or_path $pretrained_model_name_or_path \
    --target_type $target_type \
    --max_distance_json $map_json_path

export model_type="ViT-B32,ViT-B16,ViT-L14" # 攻击的模型，photomaker_clip，vae，clip，ipadapter
export pretrained_model_name_or_path="ViT-B/32,ViT-B/16,ViT-L/14"  # "/data1/humw/Pretrains/photomaker-v1.bin"，"/data1/humw/Pretrains/IP-Adapter/models/image_encoder"，"/data1/humw/Pretrains/stable-diffusion-2-1-base"
export data_dir_name="test" # 输入数据集
export w=0.75 # w=0.0, x; w=1.0, d; (1-w) * Ltgt + w * Ldevite
export attack_num=100 # 攻击轮次
export alpha=0.005 # 步长，与已有方法一致
export eps=16 # 最大噪声阈值
export input_size=512 # 输入图片的尺寸，对抗图片尺寸
export target_type="yingbu" # target图片的类型，max代表最大clip特征mse距离, yingbu代表target image为yingbu, mist代表mist图像
export device="cuda:5"
export loss_choice="cosine" # mse or cosine
echo $noise_budget_refiner

export adversarial_folder_name="${model_type}_${data_dir_name}_${loss_choice}_w${w}_num${attack_num}_alpha${alpha}_eps${eps}_input${input_size}_${target_type}";

echo $adversarial_folder_name
export adversarial_input_dir="./outputs/adversarial_images/${adversarial_folder_name}"
export customization_output_dir="./outputs/customization_outputs/${adversarial_folder_name}"
export evaluation_output_dir="./outputs/evaluation_outputs/${adversarial_folder_name}"
export map_json_path="./max_clip_cosine_distance_map.json"
# export prompts="a_photo_of_person;a_dslr_portrait_of_person" # "a_photo_of_sks_person;a_dslr_portrait_of_sks_person"
export VGGFace2="./datasets/VGGFace2"
# echo $prompts

export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/attack/ensemble_faceoff.sh" $save_config_dir

python ./attack/ensemble_faceoff.py \
    --adversarial_folder_name $adversarial_folder_name \
    --device $device \
    --prior_generation_precision "fp32" \
    --loss_choice $loss_choice \
    --w $w \
    --attack_num $attack_num \
    --alpha $alpha \
    --eps $eps \
    --input_size $input_size \
    --center_crop 1 \
    --resample_interpolation "BILINEAR" \
    --data_dir "./datasets/${data_dir_name}" \
    --input_name "set_B" \
    --data_dir_for_target_max $VGGFace2 \
    --save_dir "./outputs/adversarial_images" \
    --model_type $model_type \
    --pretrained_model_name_or_path $pretrained_model_name_or_path \
    --target_type $target_type \
    --max_distance_json $map_json_path
