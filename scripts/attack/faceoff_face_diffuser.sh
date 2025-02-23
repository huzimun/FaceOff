export model_type="face_diffuser" # target models: photomaker_clip，vae，clip，ipadapter, face_diffuser
export pretrained_model_name_or_path="/data1/humw/Pretrains/clip-vit-large-patch14"  # "/data1/humw/Pretrains/photomaker-v1.bin"，"/data1/humw/Pretrains/IP-Adapter/models/image_encoder"，"/data1/humw/Pretrains/stable-diffusion-2-1-base"
export data_dir_name="test" #validation dataset
export w=0.5 # w=0.0, x; w=1.0, d; (1-w) * Ltgt + w * Ldevite
export attack_num=50 # attack iterations
export alpha=6 # step size
export eps=16 # max noise budget
export min_eps=8 # min budget for refiner
export input_size=512 # original image size, protected image size
export model_input_size=224 # input image size for target model
export target_type="max" # type of target image: max is max clip embedding mse distance, yingbu is beijing opera mask, mist is mist image, gray is gray image
export strong_edge=200 # strong edge threshold for edge detection
export weak_edge=100 # weak edge threshold for edge detection
export mean_filter_size=3 # mean filter kernel size
export update_interval=10 # noise budget update interval
export noise_budget_refiner=0 # use refiner or not: 1 or 0 (defualt is 0)
export refiner_type="post" # post, pre, or mid0, mid1
export device="cuda:2"
export loss_choice="mse" # mse or cosine
echo $noise_budget_refiner
if [ "$noise_budget_refiner" = "1" ]; then
    export adversarial_folder_name="${model_type}_${data_dir_name}_${loss_choice}_w${w}_num${attack_num}_alpha${alpha}_eps${eps}_input${input_size}_${model_input_size}_${target_type}_refiner${noise_budget_refiner}_${refiner_type}_edge${strong_edge}-${weak_edge}_filter${mean_filter_size}_min-eps${min_eps}_interval${update_interval}";
else
    export adversarial_folder_name="${model_type}_${data_dir_name}_${loss_choice}_w${w}_num${attack_num}_alpha${alpha}_eps${eps}_input${input_size}_${model_input_size}_${target_type}_refiner${noise_budget_refiner}";
fi
echo $adversarial_folder_name
export adversarial_input_dir="./outputs/adversarial_images/${adversarial_folder_name}"
export customization_output_dir="./outputs/customization_outputs/${adversarial_folder_name}"
export evaluation_output_dir="./outputs/evaluation_outputs/${adversarial_folder_name}"
export map_json_path="/data1/humw/Codes/FaceOff/max_clip_cosine_distance_map.json"
export VGGFace2="./datasets/VGGFace2"

export save_config_dir="./outputs/config_scripts_logs/${adversarial_folder_name}"
mkdir $save_config_dir
cp "./scripts/attack/faceoff_face_diffuser.sh" $save_config_dir

python ./attack/faceoff.py \
    --device $device \
    --prior_generation_precision "bf16" \
    --loss_choice $loss_choice \
    --w $w \
    --attack_num $attack_num \
    --alpha $alpha \
    --eps $eps \
    --input_size $input_size \
    --model_input_size $model_input_size \
    --center_crop 1 \
    --resample_interpolation "BILINEAR" \
    --data_dir "./datasets/${data_dir_name}" \
    --input_name "set_B" \
    --data_dir_for_target_max $VGGFace2 \
    --save_dir "./outputs/adversarial_images" \
    --model_type $model_type \
    --pretrained_model_name_or_path $pretrained_model_name_or_path \
    --target_type $target_type \
    --max_distance_json $map_json_path \
    --min_eps $min_eps \
    --update_interval $update_interval \
    --noise_budget_refiner $noise_budget_refiner \
    --refiner_type $refiner_type \
    --mean_filter_size $mean_filter_size \
    --strong_edge $strong_edge \
    --weak_edge $weak_edge
    