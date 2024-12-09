# python ./adversarial_purification/purification.py \
#     --device "cuda:3" \
#     --data_dir './outputs/adversarial_images/ipadapter_VGGFace2_cosine_w0.5_num100_alpha6_eps16_input512_224_yingbu_refiner1_edge200-100_filter3_min-eps8_interval10' \
#     --save_dir './outputs/adversarial_images/ipadapter_VGGFace2_cosine_w0.5_num100_alpha6_eps16_input512_224_yingbu_refiner1_edge200-100_filter3_min-eps8_interval10_jpeg75' \
#     --transform_sr 0 \
#     --sr_model_path '' \
#     --transform_tvm 0 \
#     --jpeg_transform 1 \
#     --jpeg_quality 75
# python ./adversarial_purification/purification.py \
#     --device "cuda:3" \
#     --data_dir './outputs/adversarial_images/ipadapter_VGGFace2_cosine_w0.5_num100_alpha6_eps16_input512_224_yingbu_refiner0' \
#     --save_dir './outputs/adversarial_images/ipadapter_VGGFace2_cosine_w0.5_num100_alpha6_eps16_input512_224_yingbu_refiner0_sr' \
#     --transform_sr 1 \
#     --sr_model_path '/data1/humw/Pretrains/stable-diffusion-x4-upscaler' \
#     --transform_tvm 0 \
#     --jpeg_transform 0 \
#     --jpeg_quality 75
# python ./adversarial_purification/purification.py \
#     --device "cuda:3" \
#     --data_dir './outputs/adversarial_images/ipadapter_VGGFace2_cosine_w0.5_num100_alpha6_eps16_input512_224_yingbu_refiner0' \
#     --save_dir './outputs/adversarial_images/ipadapter_VGGFace2_cosine_w0.5_num100_alpha6_eps16_input512_224_yingbu_refiner0_tvm' \
#     --transform_sr 0 \
#     --sr_model_path '/data1/humw/Pretrains/stable-diffusion-x4-upscaler' \
#     --transform_tvm 1 \
#     --jpeg_transform 0 \
#     --jpeg_quality 75
python ./adversarial_purification/purification.py \
    --device "cuda:3" \
    --data_dir './outputs/adversarial_images/ipadapter_VGGFace2_cosine_w0.5_num100_alpha6_eps16_input512_224_yingbu_refiner1_edge200-100_filter3_min-eps8_interval10_time_costs' \
    --save_dir './outputs/adversarial_images/ipadapter_VGGFace2_cosine_w0.5_num100_alpha6_eps16_input512_224_yingbu_refiner1_edge200-100_filter3_min-eps8_interval10_time_costs_tvm' \
    --transform_sr 0 \
    --sr_model_path '/data1/humw/Pretrains/stable-diffusion-x4-upscaler' \
    --transform_tvm 1 \
    --jpeg_transform 0 \
    --jpeg_quality 75
python ./adversarial_purification/purification.py \
    --device "cuda:3" \
    --data_dir './outputs/adversarial_images/ipadapter_VGGFace2_cosine_w0.5_num100_alpha6_eps16_input512_224_yingbu_refiner1_edge200-100_filter3_min-eps8_interval10_time_costs' \
    --save_dir './outputs/adversarial_images/ipadapter_VGGFace2_cosine_w0.5_num100_alpha6_eps16_input512_224_yingbu_refiner1_edge200-100_filter3_min-eps8_interval10_time_costs_tvm' \
    --transform_sr 0 \
    --sr_model_path '/data1/humw/Pretrains/stable-diffusion-x4-upscaler' \
    --transform_tvm 1 \
    --jpeg_transform 0 \
    --jpeg_quality 75
python ./adversarial_purification/purification.py \
    --device "cuda:3" \
    --data_dir './outputs/adversarial_images/ipadapter_VGGFace2_cosine_w0.5_num100_alpha6_eps16_input512_224_yingbu_refiner1_edge200-100_filter3_min-eps8_interval10_time_costs' \
    --save_dir './outputs/adversarial_images/ipadapter_VGGFace2_cosine_w0.5_num100_alpha6_eps16_input512_224_yingbu_refiner1_edge200-100_filter3_min-eps8_interval10_time_costs_tvm' \
    --transform_sr 0 \
    --sr_model_path '/data1/humw/Pretrains/stable-diffusion-x4-upscaler' \
    --transform_tvm 1 \
    --jpeg_transform 0 \
    --jpeg_quality 75
python ./adversarial_purification/purification.py \
    --device "cuda:3" \
    --data_dir './outputs/adversarial_images/ipadapter_VGGFace2_cosine_w0.5_num100_alpha6_eps16_input512_224_yingbu_refiner1_edge200-100_filter3_min-eps8_interval10_time_costs' \
    --save_dir './outputs/adversarial_images/ipadapter_VGGFace2_cosine_w0.5_num100_alpha6_eps16_input512_224_yingbu_refiner1_edge200-100_filter3_min-eps8_interval10_time_costs_tvm' \
    --transform_sr 0 \
    --sr_model_path '/data1/humw/Pretrains/stable-diffusion-x4-upscaler' \
    --transform_tvm 1 \
    --jpeg_transform 0 \
    --jpeg_quality 75
python ./adversarial_purification/purification.py \
    --device "cuda:3" \
    --data_dir './outputs/adversarial_images/ipadapter_VGGFace2_cosine_w0.5_num100_alpha6_eps16_input512_224_yingbu_refiner1_edge200-100_filter3_min-eps8_interval10_time_costs' \
    --save_dir './outputs/adversarial_images/ipadapter_VGGFace2_cosine_w0.5_num100_alpha6_eps16_input512_224_yingbu_refiner1_edge200-100_filter3_min-eps8_interval10_time_costs_tvm' \
    --transform_sr 0 \
    --sr_model_path '/data1/humw/Pretrains/stable-diffusion-x4-upscaler' \
    --transform_tvm 1 \
    --jpeg_transform 0 \
    --jpeg_quality 75
