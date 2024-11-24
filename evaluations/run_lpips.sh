python ./lpips/my_lpips.py \
    --data_dir '/home/humw/Codes/FaceOff/output/Exp6/CAAT_noise-16_mini-10' \
    --emb_dirs '/home/humw/Codes/FaceOff/output/Exp6/mini-VGGFace2' \
    --save_dir '/home/humw/Codes/FaceOff/evaluations/outputs/Exp6/CAAT_noise-16_mini-10' \
    --scene 'perturbed_input' \
    --scene2 'original_input' \
    --model_name_or_path "alex" \
    --device 'cuda:5'
