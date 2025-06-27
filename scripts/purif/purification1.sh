export experiment_name="TED_agu-eot_conda-photomaker_VGGFace2_vae15-ipadapter-photomaker_mix_eot-1_yingbu_agm-2_norm-0"
export dataset_dir="./outputs/adversarial_images"

python3 ./adversarial_purification/noise.py \
    --experiment_name ${experiment_name} \
    --dataset_dir "./outputs/adversarial_images" \
    --noise 0.1

export experiment_name="TED_agu-eot_conda-photomaker_VGGFace2_vae15-ipadapter-photomaker_mix_eot-1_yingbu_agm-2_norm-0-gaussian-noise0.1"
export dataset_dir="./outputs/adversarial_images"

python3 ./adversarial_purification/purification.py \
    --device "cuda:1" \
    --experiment_name ${experiment_name} \
    --dataset_dir ${dataset_dir} \
    --transform_sr 1 \
    --sr_model_path '/data1/humw/Pretrains/stable-diffusion-x4-upscaler' \
    --transform_tvm 0 \
    --jpeg_transform 0 \
    --jpeg_quality 75


