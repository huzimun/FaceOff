import os
import torch
import torch.utils.checkpoint
import numpy as np
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
from pathlib import Path
import argparse
import mosek
import time
import os
from tqdm import tqdm
import pdb
from skimage.io import imread, imsave
from skimage.util import random_noise
from skimage import img_as_ubyte, img_as_float32
import argparse


def gaussian_noise(img, stddev):
    var = stddev * stddev
    noisy_img = random_noise(img, mode="gaussian", var=var, clip=True)
    return noisy_img

def jpeg_compress_image(image: Image.Image, quality: int = 75) -> Image.Image:
    """
    Compresses the input PIL Image object using JPEG compression and returns
    a new PIL Image object of the compressed image.
    
    :param image: PIL Image object to be compressed.
    :param quality: JPEG compression quality. Ranges from 0 to 95.
    :return: New PIL Image object of the compressed image.
    """
    compressed_image_io = BytesIO()
    image.save(compressed_image_io, 'JPEG', quality=quality)
    compressed_image_io.seek(0)  # Reset the stream position to the beginning.
    return Image.open(compressed_image_io)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example for white-box attack")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        required=True,
        help="cuda or cpu",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="test",
        required=True,
        help="test",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test",
        required=True,
        help="test",
    )
    parser.add_argument(
        "--sr_model_path",
        type=str,
        default='/data1/humw/Pretrains/stable-diffusion-x4-upscaler',
        required=True,
        help="sr_model_path",
    )
    parser.add_argument(
        "--sub_name",
        type=str,
        default='set_B',
        required=True,
        help="sub_name",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

def main(args):
    print(args)
    
    # load model and scheduler
    sr_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        args.sr_model_path, revision="fp16",torch_dtype=torch.float16) 

    sr_pipeline = sr_pipeline.to(args.device)
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # import pdb; pdb.set_trace()
    for person_id in os.listdir(input_dir):
        instance_images_path = os.path.join(input_dir, person_id, args.sub_name)
        # instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image_list = []
        for img_i_dir in os.listdir(instance_images_path):
            prompt="A photo of a person"
            img_path = os.path.join(instance_images_path, img_i_dir)
            instance_image = Image.open(img_path) # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=512x512 at 0x7E45A5F007F0>
            # import pdb; pdb.set_trace()
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
                
            # instance_image = instance_image.resize((128, 128))
            # 512 => 2048
            instance_image = sr_pipeline(image=instance_image,prompt=prompt, ).images[0]
            instance_image = instance_image.resize((512, 512), Image.Resampling.LANCZOS)
            instance_image_list.append(instance_image)
        # import pdb; pdb.set_trace()
        save_path = os.path.join(output_dir, person_id, args.sub_name)
        os.makedirs(save_path, exist_ok=True)
        img_names = [
            str(instance_path).split("/")[-1]
            for instance_path in list(Path(instance_images_path).iterdir())
        ]
        for img_pixel, img_name in zip(instance_image_list, img_names):
            if img_name.endswith(".jpg"):
                img_name = img_name.replace(".jpg", ".png")
            img_path = os.path.join(save_path, img_name)
            img_pixel.save(img_path)

    del sr_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
            
if __name__ == "__main__":
    args = parse_args()
    main(args)
