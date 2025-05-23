export input_dir='/data1/humw/Datasets/CelebA-HQ'
export output_dir='/data1/humw/Datasets/new-CelebA-HQ'

python3 ./adversarial_purification/my_sr.py \
    --device "cuda:2" \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --sr_model_path '/data1/humw/Pretrains/stable-diffusion-x4-upscaler' \
    --sub_name "set_A"

