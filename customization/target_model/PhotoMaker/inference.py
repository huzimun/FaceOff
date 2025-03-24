import torch
import numpy as np
import random
import os
from PIL import Image
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, DDIMScheduler
from huggingface_hub import hf_hub_download
from photomaker import PhotoMakerStableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
import argparse
import pdb
import json

    
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example for white-box attack")
    parser.add_argument(
        "--input_folders",
        type=str,
        default=None,
        required=True,
        help="Path to input folders",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        required=True,
        help = "input prompts"
    )
    parser.add_argument(
        "--photomaker_ckpt",
        type=str,
        default=None,
        required=True,
        help = "Path to photomaker checkpoint"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        required=True,
        help = "Path to base model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        required=True,
        help = "cpu or cuda"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        required=True,
        help = "seed"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        required=True,
        help = "num_steps"
    )
    parser.add_argument(
        "--style_strength_ratio",
        type=int,
        default=None,
        required=True,
        help = "style_strength_ratio"
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=None,
        required=True,
        help = "num_images_per_prompt"
    )
    parser.add_argument(
        "--pre_test",
        type=int,
        default=1,
        required=False,
        help = "test or not"
    )
    parser.add_argument(
        "--gaussian_filter",
        type=int,
        default=0,
        required=False,
        help = "gaussian filering after clip processor"
    )
    parser.add_argument(
        "--hflip",
        type=int,
        default=0,
        required=False,
        help = "Horizontal Flip"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='',
        required=False,
        help = "output folder"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        required=False,
        help = "height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        required=False,
        help = "width"
    )
    parser.add_argument(
        "--lora",
        type=int,
        default=0,
        required=True,
        help = "1 with lora, 0 without lora"
    )
    parser.add_argument(
        "--input_name",
        type=str,
        default="set_B",
        required=False,
        help = "input name of is_original_pipeline"
    )
    parser.add_argument(
        "--trigger_word",
        type=str,
        default="sks",
        required=False,
        help = "trigger_word"
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

def load_image_data(input_folders="", input_name="set_B"):
    input_id_images_lists = list()
    image_path_lists = list()
    for input_folder_name in sorted(os.listdir(input_folders)):
        # define and show the input ID images
        if input_name == "":
            input_folder_name = os.path.join(input_folders, input_folder_name)
        else:
            input_folder_name = os.path.join(input_folders, input_folder_name, input_name)
        image_basename_list = os.listdir(input_folder_name)
        image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in image_basename_list])

        input_id_images = []
        for image_path in image_path_list:
            input_id_images.append(load_image(image_path))
        input_id_images_lists.append(input_id_images)
        image_path_lists.append(image_path_list)
    print("Input images have been loaded !")
    return input_id_images_lists, image_path_lists

def main(args):
    print(args)
    # 1. load input images
    input_id_images_lists, image_path_lists = load_image_data(input_folders=args.input_folders, input_name=args.input_name)
    # 一次性测试所有数据耗时太长，需要考虑只测试一个人物类别的数据子集
    if args.pre_test == 1:
        image_path_lists = image_path_lists[-1:]
        input_id_images_lists = input_id_images_lists[-1:]
    
    # 2. load input prompt lists
    prompts = args.prompts.split(";")

    # 3. load model
    photomaker_ckpt = args.photomaker_ckpt
    base_model_path = args.base_model_path
    device = args.device # device = "cuda"
    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="fp16",
    ).to(device)
    print("Stable Diffusion XL have been loaded !")
    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_ckpt),
        subfolder="",
        weight_name=os.path.basename(photomaker_ckpt),
        trigger_word=args.trigger_word,
    )
    pipe.id_encoder.to(device)
    print("PhotoMaker ID Encoder have been loaded !")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # 
    if args.lora == 1:
        pipe.fuse_lora()

    generator = torch.Generator(device=device).manual_seed(args.seed) # generator = torch.Generator(device=device).manual_seed(42)
    
    # Parameter setting# generator = torch.Generator(device=device).manual_seed(42)
    # Parameter setting
    num_steps = args.num_steps # 50
    style_strength_ratio = args.style_strength_ratio # 20
    start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
    if start_merge_step > 30:
        start_merge_step = 30
    
    os.makedirs(args.save_dir, exist_ok=True)
    # 4. inference
    # pdb.set_trace()
    for i in range(0, len(input_id_images_lists)):
        for prompt in prompts:
            input_id_images = input_id_images_lists[i]
            image_path_list = image_path_lists[i]
            if args.input_name != '':
                person_id_name = image_path_list[0].split('/')[-3]
            else:
                person_id_name = image_path_list[0].split('/')[-2]
            generated_ids = [] # 如果生成终端，通过设置generated_ids来控制跳过已生成的id
            # generated_ids = ['n000061', 'n000058', 'n000057', 'n000068', 'n000076', 'n000080', 'n000063', 'n000050']
            if person_id_name in generated_ids:
                continue
            print("person_id_name:{}".format(person_id_name))
            print("prompt:{}".format(prompt))
            negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"
            images = []
            for img_idx in range(args.num_images_per_prompt):
                image = pipe(
                    prompt=prompt,
                    input_id_images=input_id_images,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=1, # 为减少显存开销，每次生成一张图像
                    num_inference_steps=num_steps,
                    start_merge_step=start_merge_step,
                    generator=generator,
                    height=args.height,
                    width=args.width,
                    gaussian_filter=args.gaussian_filter,
                    hflip=args.hflip
                ).images[0]
                images.append(image)
            # save images
            prompt_name = prompt.replace(' ', '_')
            save_folder = os.path.join(args.save_dir, person_id_name, prompt_name)
            os.makedirs(save_folder, exist_ok=True)
            # img_names = [str(instance_path).split("/")[-1] for instance_path in image_path_list]
            # for img_pixel, img_name in zip(images, img_names):
            #     save_path = os.path.join(save_folder, img_name)
            #     img_pixel.save(save_path)
            for idx, image in enumerate(images):
                save_path = os.path.join(save_folder, f"{i}_{idx}.png")
                image.save(save_path)
    del pipe

if __name__ == "__main__":
    args = parse_args()
    main(args)
