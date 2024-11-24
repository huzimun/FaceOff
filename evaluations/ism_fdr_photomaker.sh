cd deepface
pip install -e .
cd ..
cd retinaface
pip install -e .
cd ..
python ism_fdfr.py \
    --prompts 'a photo of sks person;a dslr portrait of sks person' \
    --data_dir '/home/humw/Codes/FaceOff/target_model/output/photomaker/photomaker_clip_max_out-224_res-224_stand_trans_no-mid-size_loss-n-mse_alpha6_eps16_num200_pre-test' \
    --emb_dirs '/home/humw/Datasets/VGGFace2' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/photomaker_clip_max_out-224_res-224_stand_trans_no-mid-size_loss-n-mse_alpha6_eps16_num200_pre-test' \
    --scene 'original_output' \
    --scene2 'original_input' \
    --is_target 0 \
    --map_path '/home/humw/Codes/FaceOff/target_model/PhotoMaker/VGGFace2_image_distance.json' \
    --target_path '' \
    --model_name 'ArcFace'
python ism_fdfr.py \
    --prompts 'a photo of sks person;a dslr portrait of sks person' \
    --data_dir '/home/humw/Codes/FaceOff/target_model/output/photomaker/photomaker_clip_max_out-224_res-224_stand_trans_no-mid-size_loss-n-mse_alpha6_eps16_num200_pre-test' \
    --emb_dirs '/home/humw/Datasets/VGGFace2' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/photomaker_clip_max_out-224_res-224_stand_trans_no-mid-size_loss-n-mse_alpha6_eps16_num200_pre-test' \
    --scene 'original_output' \
    --scene2 'target_input' \
    --is_target 1 \
    --map_path '/home/humw/Codes/FaceOff/target_model/PhotoMaker/VGGFace2_image_distance.json' \
    --target_path '' \
    --model_name 'ArcFace'
# 'VGG-Face' 'ArcFace'