python ./adversarial_purification/purification.py \
    --device "cuda:4" \
    --data_dir './attack/output/photomaker_clip_n_num200_alpha6_eps16_input224_output224_max_refiner0_min-eps8' \
    --save_dir './attack/output/photomaker_clip_n_num200_alpha6_eps16_input224_output224_max_refiner0_min-eps8_jpeg75' \
    --transform_sr 0 \
    --sr_model_path '' \
    --transform_tvm 0 \
    --jpeg_transform 1 \
    --jpeg_quality 75
# python ./adversarial_purification/purification.py \
#     --device "cuda:4" \
#     --data_dir '/home/humw/Codes/AAAI/FaceOff/attack/output/photomaker_clip_n_num200_alpha6_eps16_input224_output224_max_refiner0_min-eps8' \
#     --save_dir '/home/humw/Codes/AAAI/FaceOff/attack/output/photomaker_clip_n_num200_alpha6_eps16_input224_output224_max_refiner0_min-eps8_sr' \
#     --transform_sr 1 \
#     --sr_model_path '/home/humw/Pretrain/stable-diffusion-x4-upscaler' \
#     --transform_tvm 0 \
#     --jpeg_transform 0 \
#     --jpeg_quality 75
# python ./adversarial_purification/purification.py \
#     --device "cuda:4" \
#     --data_dir '/home/humw/Codes/AAAI/FaceOff/attack/output/photomaker_clip_n_num200_alpha6_eps16_input224_output224_max_refiner0_min-eps8' \
#     --save_dir '/home/humw/Codes/AAAI/FaceOff/attack/output/photomaker_clip_n_num200_alpha6_eps16_input224_output224_max_refiner0_min-eps8_tvm' \
#     --transform_sr 0 \
#     --sr_model_path '/home/humw/Pretrain/stable-diffusion-x4-upscaler' \
#     --transform_tvm 1 \
#     --jpeg_transform 0 \
#     --jpeg_quality 75