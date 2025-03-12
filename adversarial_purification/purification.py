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
        "--experiment_name",
        type=str,
        default='',
        required=True,
        help="experiment_name",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default='',
        required=True,
        help="Path to output folders",
    )
    parser.add_argument(
        "--transform_sr",
        type=int,
        default=0,
        required=True,
        help="transform_sr, 0 or 1",
    )
    parser.add_argument(
        "--sr_model_path",
        type=str,
        default='/data1/humw/Pretrains/stable-diffusion-x4-upscaler',
        required=True,
        help="sr_model_path",
    )
    parser.add_argument(
        "--transform_tvm",
        type=int,
        default=0,
        required=True,
        help="transform_tvm, 0 or 1",
    )
    parser.add_argument(
        "--jpeg_transform",
        type=int,
        default=0,
        required=True,
        help="jpeg_transform or not, 0 or 1",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=75,
        required=True,
        help="jpeg_quality 0-95",
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

def main(args):
    print(args)
    
    if args.transform_sr or args.transform_tvm:
        # load model and scheduler
        sr_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            args.sr_model_path, revision="fp16",torch_dtype=torch.float16) 

        sr_pipeline = sr_pipeline.to(args.device)
    # import pdb; pdb.set_trace()
    if args.transform_tvm:
        import cvxpy as cp
        def get_tvm_image(img, TVM_WEIGHT=0.01, PIXEL_DROP_RATE=0.02):
            # TVM_WEIGHT = 0.01
            # PIXEL_DROP_RATE = 0.02

            # Load the image
            # img = Image.open(path)  # Replace with your image path
            # reshape the image to 32*32
            # img = img.resize((64, 64), Image.ANTIALIAS)
            img = img.resize((64, 64))
            img_array = np.asarray(img, dtype=np.float32) / 255.0  # Convert to numpy array and normalize to [0, 1]

            def total_variation(Z, shape, p=2):
                h, w, c = shape
                Z_reshaped = cp.reshape(Z, (h, w*c))
                
                # Compute the Horizontal Differences and their p-norm
                horizontal_diff = Z_reshaped[1:, :] - Z_reshaped[:-1, :]
                horizontal_norm = cp.norm(horizontal_diff, p, axis=1)  # axis may need to be adjusted based on the requirement
                horizontal_sum = cp.sum(horizontal_norm)
                
                # Compute the Vertical Differences and their p-norm
                vertical_diff = Z_reshaped[:, 1:] - Z_reshaped[:, :-1]
                vertical_norm = cp.norm(vertical_diff, p, axis=0)  # axis may need to be adjusted based on the requirement
                vertical_sum = cp.sum(vertical_norm)
                
                # Total Variation is the sum of all norms
                tv = horizontal_sum + vertical_sum
                
                return tv

            def minimize_total_variation(X, x, lambda_tv=TVM_WEIGHT, p=2):
                h, w, c = x.shape
                Z = cp.Variable((h, w*c))
                X_flat = np.reshape(X, (h, w*c))
                x_flat = np.reshape(x, (h, w*c))
                objective = cp.Minimize(cp.norm(cp.multiply((1 - X_flat),(Z - x_flat)),  2) + lambda_tv * total_variation(Z, (h, w, c), p))
                problem = cp.Problem(objective)
                problem.solve(verbose=True,solver=cp.MOSEK)
                return Z.value


            # Generate the mask matrix X using Bernoulli distribution
            X = np.random.binomial(1, PIXEL_DROP_RATE, img_array.shape)

            # Run the optimization
            Z_optimal = minimize_total_variation(X, img_array)

            # reshape back to 64*64
            Z_optimal = np.reshape(Z_optimal, img_array.shape)

            # If needed, convert the result back to a PIL Imagepip install mosek
            img_result = Image.fromarray(np.uint8(Z_optimal*255))
            img_result
            return img_result
    input_dir = os.path.join(args.dataset_dir, args.experiment_name)
    output_dir = os.path.join(args.dataset_dir, args.experiment_name)
    if args.jpeg_transform:
        output_dir = output_dir + "_jpeg" + str(args.jpeg_quality)
    if args.transform_sr:
        output_dir = output_dir + "_sr"
    if args.transform_tvm:
        output_dir = output_dir + "_tvm"
    os.makedirs(output_dir, exist_ok=True)
    # import pdb; pdb.set_trace()
    for person_id in os.listdir(input_dir):
        instance_images_path = os.path.join(input_dir, person_id)
        # instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image_list = []
        for img_i_dir in os.listdir(instance_images_path):
            prompt="A photo of a person"
            img_path = os.path.join(instance_images_path, img_i_dir)
            instance_image = Image.open(img_path) # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=512x512 at 0x7E45A5F007F0>
            # import pdb; pdb.set_trace()
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            # consider some defenses like jpeg compression
            if args.jpeg_transform:
                instance_image = jpeg_compress_image(instance_image, args.jpeg_quality)
            if args.transform_sr:
                instance_image = instance_image.resize((128, 128))
                instance_image = sr_pipeline(image=instance_image,prompt=prompt, ).images[0]
            if args.transform_tvm:
                instance_image = get_tvm_image(instance_image)
                # one sr to [256, 256]
                instance_image = sr_pipeline(image=instance_image,prompt=prompt, ).images[0]
                # another resie to [128, 128]
                instance_image = instance_image.resize((128, 128))
                # another sr to [512, 512]
                instance_image = sr_pipeline(image=instance_image,prompt=prompt, ).images[0]

            instance_image_list.append(instance_image)
        # import pdb; pdb.set_trace()
        save_path = os.path.join(output_dir, person_id)
        os.makedirs(save_path, exist_ok=True)
        img_names = [
            str(instance_path).split("/")[-1]
            for instance_path in list(Path(instance_images_path).iterdir())
        ]
        for img_pixel, img_name in zip(instance_image_list, img_names):
            img_path = os.path.join(save_path, img_name)
            img_pixel.save(img_path)

    if args.transform_sr or args.transform_tvm:
        del sr_pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
if __name__ == "__main__":
    args = parse_args()
    main(args)
    # args = parse_args()
    # t1 = time.time()
    # main(args)
    # t2 = time.time()
    # print('TIME COST: %.6f'%(t2-t1))
    # with open(file="time_costs.txt", mode='a') as f:
    #     f.write(str(t2-t1) + '\n')
