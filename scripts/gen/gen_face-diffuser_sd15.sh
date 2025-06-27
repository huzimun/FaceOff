export ADV_NAME="face_diffuser,photomaker,ipadapter,ipadapterplus_min-VGGFace2_224_w1.0_num100_alpha1_eps9_input224_gau7_target-none_lpips-1_mode-no-projected"
export EXP_NAME="face-diffuser_sd15_"$ADV_NAME

export DATASET_DIR="/data1/humw/Codes/FaceOff/outputs/adversarial_images/"$ADV_NAME
export OUTPUT_DIR="/data1/humw/Codes/FaceOff/outputs/customization_outputs/"$EXP_NAME
export CAPTION="a magazine cover of a person <|image|>"
mkdir -p $OUTPUT_DIR

cd ./customization/target_model/Face-diffuser
for person_id in `ls $DATASET_DIR`; do
    export INSTANCE_DIR=${DATASET_DIR}"/"${person_id}"/"
    export FACE_OUTPUT_DIR=$OUTPUT_DIR"/"${person_id}
    echo ${INSTANCE_DIR}
    echo ${FACE_OUTPUT_DIR}
    mkdir -p ${FACE_OUTPUT_DIR}

    CUDA_VISIBLE_DEVICES=1 python3 ./facediffuser/inference.py \
        --pretrained_model_name_or_path /data1/humw/Pretrains/stable-diffusion-v1-5  \
        --finetuned_model_path model/SDM \
        --finetuned_model_text_only_path model/TDM \
        --test_reference_folder ${INSTANCE_DIR} \
        --person_id "${person_id}" \
        --test_caption "${CAPTION}" \
        --output_dir ${FACE_OUTPUT_DIR} \
        --mixed_precision fp16 \
        --image_encoder_type clip \
        --image_encoder_name_or_path /data1/humw/Pretrains/clip-vit-large-patch14 \
        --num_image_tokens 1 \
        --max_num_objects 1 \
        --object_resolution 512 \
        --generate_height 512 \
        --generate_width 512 \
        --num_images_per_prompt 16 \
        --num_rows 1 \
        --seed 3407\
        --guidance_scale 5 \
        --inference_steps 50 \
        --start_merge_step 15 \
        --final_step 30 \
        --no_object_augmentation
done



export ADV_NAME="face_diffuser,photomaker,ipadapter,ipadapterplus_min-VGGFace2_224_w1.0_num100_alpha1_eps9_input224_gau7_target-none_l1-0_affine-0_mode-idprotector"
export EXP_NAME="face-diffuser_sd15_"$ADV_NAME

export DATASET_DIR="/data1/humw/Codes/FaceOff/outputs/adversarial_images/"$ADV_NAME
export OUTPUT_DIR="/data1/humw/Codes/FaceOff/outputs/customization_outputs/"$EXP_NAME
export CAPTION="a magazine cover of a person <|image|>"
mkdir -p $OUTPUT_DIR

cd ./customization/target_model/Face-diffuser
for person_id in `ls $DATASET_DIR`; do
    export INSTANCE_DIR=${DATASET_DIR}"/"${person_id}"/"
    export FACE_OUTPUT_DIR=$OUTPUT_DIR"/"${person_id}
    echo ${INSTANCE_DIR}
    echo ${FACE_OUTPUT_DIR}
    mkdir -p ${FACE_OUTPUT_DIR}

    CUDA_VISIBLE_DEVICES=1 python3 ./facediffuser/inference.py \
        --pretrained_model_name_or_path /data1/humw/Pretrains/stable-diffusion-v1-5  \
        --finetuned_model_path model/SDM \
        --finetuned_model_text_only_path model/TDM \
        --test_reference_folder ${INSTANCE_DIR} \
        --person_id "${person_id}" \
        --test_caption "${CAPTION}" \
        --output_dir ${FACE_OUTPUT_DIR} \
        --mixed_precision fp16 \
        --image_encoder_type clip \
        --image_encoder_name_or_path /data1/humw/Pretrains/clip-vit-large-patch14 \
        --num_image_tokens 1 \
        --max_num_objects 1 \
        --object_resolution 512 \
        --generate_height 512 \
        --generate_width 512 \
        --num_images_per_prompt 16 \
        --num_rows 1 \
        --seed 3407\
        --guidance_scale 5 \
        --inference_steps 50 \
        --start_merge_step 15 \
        --final_step 30 \
        --no_object_augmentation
done
