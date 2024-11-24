python ./resize_to_224_image.py \
    --data_dir '/home/humw/Codes/FaceOff/output/Exp1/photomaker/ours_ln_refiner_mini-10' \
    --save_dir '/home/humw/Codes/FaceOff/output/Exp1/photomaker/ours_ln_refiner_mini-10_224' \
    --sub_folder '' \
    --resolution 224
python ./LIQE/run_liqe_for_input.py \
    --data_dir '/home/humw/Codes/FaceOff/output/Exp1/photomaker/ours_ln_refiner_mini-10_224' \
    --sub_folder '' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/Exp1/photomaker/ours_ln_refiner_mini-10' \
    --scene 'original_input_224' \
    --device 'cuda:7'
python ./resize_to_224_image.py \
    --data_dir '/home/humw/Codes/FaceOff/output/Exp1/photomaker/ours_lx_refiner_mini-10' \
    --save_dir '/home/humw/Codes/FaceOff/output/Exp1/photomaker/ours_lx_refiner_mini-10_224' \
    --sub_folder '' \
    --resolution 224
python ./LIQE/run_liqe_for_input.py \
    --data_dir '/home/humw/Codes/FaceOff/output/Exp1/photomaker/ours_lx_refiner_mini-10_224' \
    --sub_folder '' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/Exp1/photomaker/ours_lx_refiner_mini-10' \
    --scene 'original_input_224' \
    --device 'cuda:7'