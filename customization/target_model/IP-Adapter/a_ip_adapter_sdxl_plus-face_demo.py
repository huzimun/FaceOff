import torch
from PIL import Image
import os
from ip_adapter import IPAdapterPlusXL
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline
from pathlib import Path
import argparse

def load_data(data_dir, image_size=224, resample=2):
    import numpy as np
    def image_to_numpy(image):
        return np.array(image).astype(np.uint8)
    # more robust loading to avoid loaing non-image files
    images = [] 
    for i in sorted(list(Path(data_dir).iterdir())):
        if not i.suffix in [".jpg", ".png", ".jpeg"]:
            continue
        else:
            images.append(image_to_numpy(Image.open(i).convert("RGB")))
    # resize the images to 512 x 512, resample value 2 means BILINEAR
    images = [Image.fromarray(i).resize((image_size, image_size), resample) for i in images]
    return images

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example for white-box attack")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        required=False,
        help="Path to input folders",
    )
    parser.add_argument(
        "--sub_name",
        type=str,
        default="set_B",
        required=True,
        help=("subfolder name of input dir"),
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/home/humw/Pretrain/RealVisXL_V3.0",
        required=False,
        help=("/home/humw/Pretrain/RealVisXL_V3.0"),
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="/home/humw/Pretrain/h94/IP-Adapter/models/image_encoder",
        required=False,
        help = "hflip or not"
    )
    parser.add_argument(
        "--ip_ckpt",
        type=str,
        default="/home/humw/Pretrain/h94/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin",
        required=False,
        help = "ip ckpt"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/humw/Codes/FaceOff/output/Exp1/ipadapter/min-VGGFace2_ipadapter_out-224_no-mid-size_loss-n-mse_alpha6_eps16_num200_pre-test",
        required=False,
        help = "input dir"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/humw/Codes/FaceOff/target_model/output/ipadapter/min-VGGFace2_ipadapter_out-224_no-mid-size_loss-n-mse_alpha6_eps16_num200_pre-test",
        required=False,
        help = "output dir"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        required=False,
        help = "image resolution of target model, clip is 224"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

def main(args):
    print(args)
    # load SDXL pipeline
    pipe = StableDiffusionXLCustomPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )

    # load ip-adapter
    ip_model = IPAdapterPlusXL(pipe, args.image_encoder_path, args.ip_ckpt, args.device, num_tokens=16)
    for person_id in sorted(os.listdir(args.input_dir)):
        print(person_id)
        person_dir = os.path.join(args.input_dir, person_id + "/" + args.sub_name)
        print(person_dir)
        images = load_data(person_dir, args.resolution, 2)
        output_images1 = list()
        output_images2 = list()
        for image in images:
            output_images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=30, seed=42, prompt="a photo of person")
            output_images1 += output_images
        for image in images:
            output_images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=30, seed=42, prompt="a dslr portrait of person")
            output_images2 += output_images
        save_path = os.path.join(args.output_dir, person_id)
        os.makedirs(save_path, exist_ok=True)
        prompt1_path = "a_photo_of_person"
        save_prompt_path = os.path.join(save_path, prompt1_path)
        os.makedirs(save_prompt_path, exist_ok=True)
        for idx, image in enumerate(output_images1):
            image.save(os.path.join(save_prompt_path, f"ipadapter_{idx:02d}.png"))
        prompt2_path = "a_dslr_portrait_of_person"
        save_prompt_path = os.path.join(save_path, prompt2_path)
        os.makedirs(save_prompt_path, exist_ok=True)
        for idx, image in enumerate(output_images2):
            image.save(os.path.join(save_prompt_path, f"ipadapter_{idx:02d}.png"))

if __name__ == "__main__":
    args = parse_args()
    main(args)
