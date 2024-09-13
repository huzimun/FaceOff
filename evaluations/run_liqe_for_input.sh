python ./LIQE/run_liqe_for_input.py \
    --data_dir '/home/humw/Datasets/CelebA-HQ' \
    --sub_folder 'set_B' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/CelebA-HQ_original_output' \
    --scene 'original_input' \
    --device 'cuda:7'
python ./LIQE/run_liqe_for_input.py \
    --data_dir '/home/humw/Datasets/VGGFace2' \
    --sub_folder 'set_B' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/VGGFace2_original_output' \
    --scene 'original_input' \
    --device 'cuda:7'
python ./LIQE/run_liqe_for_input.py \
    --data_dir '/home/humw/Codes/FaceOff/output/VGGFace2_photomaker_clip_max_out-224_res-224_stand_trans_no-refiner_loss-x-mse_alpha6_eps16_num200' \
    --sub_folder '' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/VGGFace2_photomaker_clip_max_out-224_res-224_stand_trans_no-refiner_loss-x-mse_alpha6_eps16_num200' \
    --scene 'protected_input' \
    --device 'cuda:7'
python ./LIQE/run_liqe_for_input.py \
    --data_dir '/home/humw/Codes/FaceOff/output/VGGFace2_photomaker_clip_max_out-224_res-224_stand_trans_no-refiner_loss-n-mse_alpha6_eps16_num200' \
    --sub_folder '' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/VGGFace2_photomaker_clip_max_out-224_res-224_stand_trans_no-refiner_loss-n-mse_alpha6_eps16_num200' \
    --scene 'protected_input' \
    --device 'cuda:7'
python ./LIQE/run_liqe_for_input.py \
    --data_dir '/home/humw/Codes/FaceOff/output/CelebA-HQ_photomaker_clip_max_out-224_res-224_stand_trans_no-refiner_loss-x-mse_alpha6_eps16_num200' \
    --sub_folder '' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/CelebA-HQ_photomaker_clip_max_out-224_res-224_stand_trans_no-refiner_loss-x-mse_alpha6_eps16_num200' \
    --scene 'protected_input' \
    --device 'cuda:7'
python ./LIQE/run_liqe_for_input.py \
    --data_dir '/home/humw/Codes/FaceOff/output/CelebA-HQ_photomaker_clip_max_out-224_res-224_stand_trans_no-refiner_loss-n-mse_alpha6_eps16_num200' \
    --sub_folder '' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/CelebA-HQ_photomaker_clip_max_out-224_res-224_stand_trans_no-refiner_loss-n-mse_alpha6_eps16_num200' \
    --scene 'protected_input' \
    --device 'cuda:7'