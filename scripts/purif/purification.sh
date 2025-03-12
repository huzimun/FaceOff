export experiment_name="ASPL_ace_VGGFace2_SD15_mist"
export dataset_dir="./outputs/adversarial_images"

python3 ./adversarial_purification/purification.py \
    --device "cuda:0" \
    --experiment_name ${experiment_name} \
    --dataset_dir ${dataset_dir} \
    --transform_sr 0 \
    --sr_model_path '' \
    --transform_tvm 0 \
    --jpeg_transform 1 \
    --jpeg_quality 75

# python3 ./adversarial_purification/purification.py \
#     --device "cuda:0" \
#     --experiment_name ${experiment_name} \
#     --dataset_dir ${dataset_dir} \
#     --transform_sr 1 \
#     --sr_model_path '/data1/humw/Pretrains/stable-diffusion-x4-upscaler' \
#     --transform_tvm 0 \
#     --jpeg_transform 0 \
#     --jpeg_quality 75

# python3 ./adversarial_purification/noise.py \
#     --experiment_name "ASPL_ace_VGGFace2_SD15_mist" \
#     --dataset_dir "./outputs/adversarial_images" \
#     --noise 0.1

# python3 ./adversarial_purification/purification.py \
#     --device "cuda:0" \
#     --experiment_name ${experiment_name} \
#     --dataset_dir ${dataset_dir} \
#     --transform_sr 0 \
#     --sr_model_path '/data1/humw/Pretrains/stable-diffusion-x4-upscaler' \
#     --transform_tvm 1 \
#     --jpeg_transform 0 \
#     --jpeg_quality 75