export model_type='photomaker_clip' # 攻击的模型，photomaker_clip，vae，clip，ipadapter
export pretrained_model_name_or_path="/data1/humw/Pretrains/photomaker-v1.bin"  # "/data1/humw/Pretrains/photomaker-v1.bin"，'/data1/humw/Pretrains/IP-Adapter/models/image_encoder'，'/data1/humw/Pretrains/stable-diffusion-2-1-base'
export data_dir_name="VGGFace2" # 输入数据集
export w=0 # w=0, x; w=1, d; (1-w) * Ltgt + w * Ldevite
export attack_num=100 # 攻击轮次
export alpha=6 # 步长
export eps=16 # 最大噪声阈值
export min_eps=8 # refiner的最小噪声阈值
export input_size=512 # 输入图片的尺寸，对抗图片尺寸
export model_input_size=224 # 模型输入图片尺寸
export target_type="yingbu" # target图片的类型，max代表最大clip特征mse距离, yingbu代表target image为yingbu
export strong_edge=200 # 边缘检测器的强边
export weak_edge=100 # 边缘检测器的弱边
export mean_filter_size=3 # 均值滤波器的尺寸
export update_interval=10 # 阈值更新间隔
export noise_budget_refiner=0 # 是否使用refiner
export device="cuda:2"
echo $noise_budget_refiner
if [ $noise_budget_refiner == 1 ];then
    export adversarial_folder_name="${model_type}_${data_dir_name}_w${w}_num${attack_num}_alpha${alpha}_eps${eps}_input${input_size}_${model_input_size}_${target_type}_refiner${noise_budget_refiner}_edge${strong_edge}-${weak_edge}_filter${mean_filter_size}_min-eps${min_eps}_interval${update_interval}";
else
    export adversarial_folder_name="${model_type}_${data_dir_name}_w${w}_num${attack_num}_alpha${alpha}_eps${eps}_input${input_size}_${model_input_size}_${target_type}_refiner${noise_budget_refiner}";
fi
echo $adversarial_folder_name
export adversarial_input_dir="./outputs/photomaker/adversarial_images/${adversarial_folder_name}"
export customization_output_dir="./outputs/photomaker/customization_outputs/${adversarial_folder_name}"
export evaluation_output_dir="./outputs/photomaker/evaluation_outputs/${adversarial_folder_name}"
export original_output_dir="./outputs/photomaker/customization_outputs/mini-VGGFace2/"
export map_json_path="./customization/target_model/PhotoMaker/VGGFace2_max_photomaker_clip_distance.json"
export prompts="a_photo_of_sks_person;a_dslr_portrait_of_sks_person" # "a_photo_of_sks_person;a_dslr_portrait_of_sks_person"
export VGGFace2="./datasets/VGGFace2"
echo $prompts

export save_config_dir="./outputs/photomaker/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/attack_gen_eval_photomaker.sh" $save_config_dir

python ./attack/faceoff.py \
    --device $device \
    --prior_generation_precision "bf16" \
    --w $w \
    --attack_num $attack_num \
    --alpha $alpha \
    --eps $eps \
    --input_size $input_size \
    --model_input_size $model_input_size \
    --center_crop 1 \
    --resample_interpolation 'BILINEAR' \
    --data_dir "./datasets/${data_dir_name}" \
    --input_name "set_B" \
    --data_dir_for_target_max $VGGFace2 \
    --save_dir "./outputs/photomaker/adversarial_images" \
    --model_type $model_type \
    --pretrained_model_name_or_path $pretrained_model_name_or_path \
    --target_type $target_type \
    --max_distance_json $map_json_path \
    --min_eps $min_eps \
    --update_interval $update_interval \
    --noise_budget_refiner $noise_budget_refiner \
    --mean_filter_size $mean_filter_size \
    --strong_edge $strong_edge \
    --weak_edge $weak_edge

# attack PhotoMaker
python ./customization/target_model/PhotoMaker/inference.py \
    --input_folders "./outputs/photomaker/adversarial_images/${adversarial_folder_name}" \
    --save_dir "./outputs/photomaker/customization_outputs/${adversarial_folder_name}" \
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

