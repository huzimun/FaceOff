python ./my_clip/my_clip.py \
    --prompts 'a photo of sks person;a dslr portrait of sks person' \
    --data_dir '/home/humw/Codes/FaceOff/target_model/output/photomaker/VGGFace2_original_output' \
    --emb_dirs '/home/humw/Datasets/VGGFace2' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/VGGFace2_original_output' \
    --scene 'original_output' \
    --scene2 'original_input' \
    --is_target 0 \
    --map_path '/home/humw/Codes/FaceOff/target_model/PhotoMaker/VGGFace2_image_distance.json' \
    --target_path '' \
    --model_name_or_path 'ViT-B/32' \
    --device 'cuda:5'
# 'VGG-Face' 'ArcFace'