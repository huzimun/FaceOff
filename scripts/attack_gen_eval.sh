# python ./target_model/IP-Adapter/a_ip_adapter_sdxl_plus-face_demo.py \
#     --base_model_path "/home/humw/Pretrain/RealVisXL_V3.0" \
#     --device "cuda:7" \
#     --image_encoder_path "/home/humw/Pretrain/h94/IP-Adapter/models/image_encoder" \
#     --ip_ckpt "/home/humw/Pretrain/h94/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin" \
#     --input_dir "/home/humw/Codes/FaceOff/output/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200/" \
#     --output_dir "/home/humw/Codes/FaceOff/target_model/output/ipadapter/Exp2/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200" \
#     --resolution 224 \
#     --sub_name ""
cd evaluations
# cd deepface
# pip install -e .
# cd ..
# cd retinaface
# pip install -e .
# cd ..

# 'a photo of person;a dslr portrait of person'

python ism_fdfr.py \
    --prompts 'a photo of person;a dslr portrait of person' \
    --data_dir "/home/humw/Codes/FaceOff/target_model/output/ipadapter/Exp2/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200" \
    --emb_dirs '/home/humw/Datasets/VGGFace2' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/Exp2/ipadapter/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200' \
    --scene 'protected_output' \
    --scene2 'original_input' \
    --is_target 0 \
    --map_path '/home/humw/Codes/FaceOff/VGGFace2_image_distance.json' \
    --target_path '' \
    --model_name 'ArcFace'

python ism_fdfr.py \
    --prompts 'a photo of person;a dslr portrait of person' \
    --data_dir "/home/humw/Codes/FaceOff/target_model/output/ipadapter/Exp2/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200" \
    --emb_dirs '/home/humw/Datasets/VGGFace2' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/Exp2/ipadapter/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200' \
    --scene 'protected_output' \
    --scene2 'target_input' \
    --is_target 1 \
    --map_path '/home/humw/Codes/FaceOff/VGGFace2_image_distance.json' \
    --target_path '' \
    --model_name 'ArcFace'

python ./my_clip/my_clip.py \
    --prompts 'a photo of person;a dslr portrait of person' \
    --data_dir "/home/humw/Codes/FaceOff/target_model/output/ipadapter/Exp2/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200" \
    --emb_dirs '/home/humw/Datasets/VGGFace2' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/Exp2/ipadapter/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200' \
    --scene 'protected_output' \
    --scene2 'original_input' \
    --is_target 0 \
    --map_path '/home/humw/Codes/FaceOff/VGGFace2_image_distance.json' \
    --target_path '' \
    --model_name_or_path 'ViT-B/32' \
    --device 'cuda:7'

python ./my_clip/my_clip.py \
    --prompts 'a photo of person;a dslr portrait of person' \
    --data_dir "/home/humw/Codes/FaceOff/target_model/output/ipadapter/Exp2/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200" \
    --emb_dirs '/home/humw/Datasets/VGGFace2' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/Exp2/ipadapter/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200' \
    --scene 'protected_output' \
    --scene2 'target_input' \
    --is_target 1 \
    --map_path '/home/humw/Codes/FaceOff/VGGFace2_image_distance.json' \
    --target_path '' \
    --model_name_or_path 'ViT-B/32' \
    --device 'cuda:7'

python ./LIQE/run_liqe.py \
    --prompts 'a photo of person;a dslr portrait of person' \
    --data_dir "/home/humw/Codes/FaceOff/target_model/output/ipadapter/Exp2/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200" \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/Exp2/ipadapter/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200' \
    --scene 'protected_output' \
    --device 'cuda:7'
python ./resize_to_224_image.py \
    --data_dir "/home/humw/Codes/FaceOff/output/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200/" \
    --save_dir "/home/humw/Codes/FaceOff/output/Exp2/ipadapter/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200_224" \
    --sub_folder '' \
    --resolution 224
python ./LIQE/run_liqe_for_input.py \
    --data_dir "/home/humw/Codes/FaceOff/output/Exp2/ipadapter/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200_224" \
    --sub_folder '' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/Exp2/ipadapter/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200' \
    --scene 'protected_input' \
    --device 'cuda:7'
python ./lpips/my_lpips.py \
    --data_dir "/home/humw/Codes/FaceOff/output/Exp2/ipadapter/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200_224" \
    --emb_dirs '/home/humw/Codes/FaceOff/output/Exp6/mini-VGGFace2' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/Exp2/ipadapter/new_min-VGGFace2_ipadapter_out-224_loss-x-mse_alpha6_eps16_num200' \
    --scene 'perturbed_input' \
    --scene2 'original_input' \
    --model_name_or_path "alex" \
    --device 'cuda:7' \
    --resolution 224
