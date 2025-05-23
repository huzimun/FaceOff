export EXPERIMENT_NAME="TED"
export device="cuda:0"

export DATASET="VGGFace2"
if [ "$DATASET" = "VGGFace2" ]; then
    export DATASET_DIR="/data1/xxxx/Datasets/VGGFace2"
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${DATASET}
else
    echo "Invalid DATASET"
    exit 1
fi

export model_types="vae15-ipadapter-photomaker"
EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${model_types}

# distance choice for adv attack loss
export distance_choice="mix" # mse or cosine, mix use mse for vae, and cosine for ipadapter and photomaker
if [ "$distance_choice" = "mse" ] || [ "$distance_choice" = "cosine" ] || [ "$distance_choice" = "mix" ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${distance_choice}
else
    echo "Invalid distance_choice"
    exit 1
fi

export eot=0
if [ "$eot" = 1 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_eot-1"
else
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_eot-0"
fi

export target="yingbu" # "yingbu" "mist" "max-mask" "min-mask" "random-mask"
export id_map_path="test" # map original id to target id
EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${target}
if [ "$target" = "mist" ]; then
    export target_image_path="./target_images/mist"
elif [ "$target" = "yingbu" ]; then
    export target_image_path="./target_images/yingbu"
elif [ "$target" = "non-target" ]; then
    export target_image_path="non-target"
else
    echo "Invalid target"
    exit 1
fi


export save_config_dir="./outputs/config_scripts_logs/${EXPERIMENT_NAME}"
mkdir $save_config_dir
cp "./scripts/attack/ted.sh" $save_config_dir

for person_id in `ls $DATASET_DIR`; do   
    export CLEAN_ADV_DIR=${DATASET_DIR}"/"${person_id}"/set_B"
    export ADV_OUTPUT_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME"/"${person_id}
    echo ${CLEAN_ADV_DIR}
    echo ${ADV_OUTPUT_DIR}
    # ------------------------- Train ASPL on set B -------------------------
    mkdir -p $ADV_OUTPUT_DIR
    
    # Generate Protecting Images
    python3 attacks/ted.py \
        --model_types $model_types \
        --device=$device \
        --seed=1 \
        --eot $eot \
        --target $target \
        --distance_choice $distance_choice \
        --target_image_path $target_image_path \
        --pretrained_model_name_or_path=$MODEL_PATH  \
        --mixed_precision "bf16" \
        --enable_xformers_memory_efficient_attention \
        --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
        --output_dir=$ADV_OUTPUT_DIR \
        --center_crop \
        --resolution=512 \
        --max_train_steps=50 \
        --max_adv_train_steps=6 \
        --pgd_alpha=5e-3 \
        --pgd_eps=0.12549019607843137
        
done 
