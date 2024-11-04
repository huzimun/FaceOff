python ./customization/target_model/IP-Adapter/a_ip_adapter_sdxl_plus-face_demo.py \
    --base_model_path "/home/humw/Pretrain/RealVisXL_V3.0" \
    --image_encoder_path "/home/humw/Pretrain/h94/IP-Adapter/models/image_encoder" \
    --ip_ckpt "/home/humw/Pretrain/h94/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin" \
    --device "cuda:1" \
    --input_dir "/home/humw/Datasets/mini-VGGFace2" \
    --output_dir "/home/humw/Codes/FaceOff/output/ipadapter/customization_outputs/mini-VGGFace2" \
    --resolution 224 \
    --sub_name "set_B"