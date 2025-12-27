import gradio as gr
import torch
import torch.nn.functional as F
import clip
from torchvision import transforms
import os
from PIL import Image
import json
from pathlib import Path
import tempfile
import numpy as np
import time
from typing import List, Dict, Any
import argparse
import glob
import cv2
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
from diffusers import AutoencoderKL
import sys
sys.path.append('/home/humw/Codes/FaceOff/gradio/ip_adapter')
from ip_adapter.resampler import Resampler
from ip_adapter.ip_adapter import ImageProjModel

# 设置随机种子
seed = 1
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class ImageProtector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dict = {}
        self.image_proj_model_dict = {}
        self.eot_trans_list = []
        
    def setup_models(self, model_types: List[str], precision: str = "fp16"):
        """设置模型"""
        if precision == "fp32":
            torch_dtype = torch.float32
        elif precision == "fp16":
            torch_dtype = torch.float16
        elif precision == "bf16":
            torch_dtype = torch.bfloat16
        else:
            raise ValueError("precision must be one of [fp32, fp16, bf16]")
            
        self.model_dict = {}
        self.image_proj_model_dict = {}
        
        # 根据模型类型设置默认的模型路径
        model_paths = {
            'clip': '/data1/humw/Pretrains/ViT-B-32.pt',
            'photomaker': '/data1/humw/Pretrains/photomaker-v1.bin',
            'ipadapter': '/data1/humw/Pretrains/IP-Adapter/models/image_encoder',
            'ipadapter-plus': '/data1/humw/Pretrains/IP-Adapter/sdxl_models/image_encoder',
            'face_diffuser': '/data1/humw/Pretrains/clip-vit-large-patch14'
        }
        
        for model_type in model_types:
            if model_type in ['clip', 'ViT-B32', 'ViT-B16', 'ViT-L14']:
                model, _ = clip.load(model_paths['clip'], device=self.device)
                model.to(torch_dtype)
                self.model_dict[model_type] = model
                
            elif model_type == 'photomaker':
                # 简化版的PhotoMaker模型加载
                try:
                    from photomaker_clip import PhotoMakerIDEncoder
                    model = PhotoMakerIDEncoder()
                    # 这里应该加载预训练权重，简化处理
                    model.to(self.device, dtype=torch_dtype)
                    self.model_dict[model_type] = model
                except:
                    print(f"Warning: Could not load {model_type} model")
                    
            elif model_type in ['ipadapter', 'ipadapter-plus']:
                try:
                    model = CLIPVisionModelWithProjection.from_pretrained(
                        model_paths['ipadapter']).to(self.device, dtype=torch_dtype)
                    self.model_dict[model_type] = model
                    # 简化投影模型
                    self.image_proj_model_dict[model_type] = None
                except:
                    print(f"Warning: Could not load {model_type} model")
                    
            elif model_type == 'face_diffuser':
                try:
                    # 简化版的FaceDiffuser模型
                    class SimpleFaceDiffuser(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
                            
                        def forward(self, x):
                            return self.encoder(x).image_embeds
                    
                    model = SimpleFaceDiffuser().to(self.device, dtype=torch_dtype)
                    self.model_dict[model_type] = model
                except:
                    print(f"Warning: Could not load {model_type} model")
        
        return True
    
    def setup_transforms(self, eot_trans_types: List[str], resample_interpolation: str = "BILINEAR"):
        """设置变换"""
        if resample_interpolation == 'BILINEAR':
            resample_interpolation_enum = transforms.InterpolationMode.BILINEAR
        else:
            resample_interpolation_enum = transforms.InterpolationMode.BICUBIC
            
        self.eot_trans_list = []
        
        for trans_type in eot_trans_types:
            train_aug_for_clip = [
                transforms.Resize(224, interpolation=resample_interpolation_enum),
                transforms.CenterCrop(224),
            ]
            
            tensorize_and_normalize = [
                transforms.Normalize([0.5 * 255]*3, [0.5 * 255]*3),
            ]
            
            if trans_type == 'gau':
                gau_filter = transforms.GaussianBlur(kernel_size=7)
                defense_transform = [gau_filter]
            elif trans_type == 'hflip':
                hflip = transforms.RandomHorizontalFlip(p=0.5)
                defense_transform = [hflip]
            elif trans_type == 'none':
                defense_transform = []
            elif trans_type == 'gau-hflip':
                gau_filter = transforms.GaussianBlur(kernel_size=7)
                hflip = transforms.RandomHorizontalFlip(p=0.5)
                defense_transform = [gau_filter, hflip]
            else:
                defense_transform = []
                
            trans_for_clip = train_aug_for_clip + defense_transform + tensorize_and_normalize
            trans_for_clip = transforms.Compose(trans_for_clip)
            self.eot_trans_list.append(trans_for_clip)
            
        return True
    
    def load_target_data(self, target_type: str, input_size: int):
        """加载目标数据"""
        # 创建简单的目标图像
        if target_type == 'max':
            # 创建随机纹理作为目标
            target_array = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
        elif target_type == 'gray':
            target_array = np.full((input_size, input_size, 3), 128, dtype=np.uint8)
        elif target_type == 'mist':
            # 创建雾状效果
            target_array = np.random.randint(200, 255, (input_size, input_size, 3), dtype=np.uint8)
        elif target_type == 'colored_mist':
            # 创建彩色雾状效果
            target_array = np.random.randint(150, 255, (input_size, input_size, 3), dtype=np.uint8)
        else:  # yingbu or default
            # 创建京剧脸谱风格的底色
            base_color = [200, 150, 100]  # 橙黄色调
            target_array = np.full((input_size, input_size, 3), base_color, dtype=np.uint8)
        
        target_image = Image.fromarray(target_array)
        target_tensor = torch.from_numpy(np.array(target_image)).permute(2, 0, 1).unsqueeze(0).float()
        return target_tensor
    
    def pgd_ensemble_attack(self, perturbed_data, origin_data, alpha, eps, attack_num, 
                           target_data, loss_choice, w):
        """PGD集成攻击算法 - 从文档2移植"""
        with torch.no_grad():
            origin_data.requires_grad_(False)
            eot_trans_target_data_list = []
            eot_trans_origin_data_list = []
            # print("计算EOT变换下的目标数据和原始数据")
            for trans in self.eot_trans_list:
                tran_target_data = trans(target_data)
                tran_origin_data = trans(origin_data)
                eot_trans_target_data_list.append(tran_target_data)
                eot_trans_origin_data_list.append(tran_origin_data)
            
            d_type = perturbed_data.dtype
            # 初始扰动
            # print("添加初始扰动")
            perturbed_data = (perturbed_data + (torch.rand(*perturbed_data.shape)*2*eps-eps).to(perturbed_data.device)).to(d_type)
            
            target_embeds_dict = {}
            origin_embeds_dict = {}
            
            # 计算目标嵌入和原始嵌入
            # print("计算目标嵌入和原始嵌入")
            for k in self.model_dict.keys():
                model_type = k
                model = self.model_dict[k]
                eot_trans_target_embeds_list = []
                eot_trans_origin_embeds_list = []
                # print(f"Processing model: {model_type}")
                for i in range(len(self.eot_trans_list)):
                    # print(f"  EOT transform {i+1}/{len(self.eot_trans_list)}")
                    tran_target_data = eot_trans_target_data_list[i]
                    tran_origin_data = eot_trans_origin_data_list[i]
                    # print(type(tran_target_data), tran_target_data.shape)
                    # print(type(tran_origin_data), tran_origin_data.shape)
                    # import pdb; pdb.set_trace()
                    if model_type in ['clip', 'ViT-B32', 'ViT-B16', 'ViT-L14']:
                        target_embeds = model.encode_image(tran_target_data) # 
                        origin_embeds = model.encode_image(tran_origin_data)
                    elif model_type == 'photomaker':
                        target_embeds = model(tran_target_data)
                        origin_embeds = model(tran_origin_data)
                    elif model_type in ['ipadapter', 'ipadapter-plus']:
                        target_embeds = model(tran_target_data).image_embeds
                        origin_embeds = model(tran_origin_data).image_embeds
                    elif model_type == 'face_diffuser':
                        target_embeds = model(tran_target_data.unsqueeze(0))
                        origin_embeds = model(tran_origin_data.unsqueeze(0))
                    else:
                        # # 默认使用CLIP-like编码
                        # target_embeds = model.encode_image(tran_target_data) if hasattr(model, 'encode_image') else model(tran_target_data)
                        # origin_embeds = model.encode_image(tran_origin_data) if hasattr(model, 'encode_image') else model(tran_origin_data)
                        raise ValueError(f"Unknown model type: {model_type}")
                    
                    eot_trans_target_embeds_list.append(target_embeds)
                    eot_trans_origin_embeds_list.append(origin_embeds)
                
                target_embeds_dict[k] = eot_trans_target_embeds_list
                origin_embeds_dict[k] = eot_trans_origin_embeds_list
        
        Loss_dict = {}
        
        # PGD攻击迭代
        # print("开始PGD攻击迭代")
        for epoch in range(attack_num):
            perturbed_data.requires_grad_()
            eot_trans_perturbed_data_list = []
            
            for trans in self.eot_trans_list:
                tran_perturbed_data = trans(perturbed_data)
                eot_trans_perturbed_data_list.append(tran_perturbed_data)
            
            Loss_x_ = []
            Loss_d_ = []
            # print("计算损失")
            for k in self.model_dict.keys():
                model_type = k
                model = self.model_dict[k]
                loss_x_list = []
                loss_d_list = []
                
                for i in range(len(self.eot_trans_list)):
                    tran_perturbed_data = eot_trans_perturbed_data_list[i]
                    wi = 1  # 权重参数
                    
                    if model_type in ['clip', 'ViT-B32', 'ViT-B16', 'ViT-L14']:
                        adv_embeds = model.encode_image(tran_perturbed_data)
                    elif model_type == 'photomaker':
                        adv_embeds = model(tran_perturbed_data)
                    elif model_type in ['ipadapter', 'ipadapter-plus']:
                        adv_embeds = model(tran_perturbed_data).image_embeds
                    elif model_type == 'face_diffuser':
                        adv_embeds = model(tran_perturbed_data.unsqueeze(0))
                    else:
                        adv_embeds = model.encode_image(tran_perturbed_data) if hasattr(model, 'encode_image') else model(tran_perturbed_data)
                    
                    target_embeds_list = target_embeds_dict[k]
                    origin_embeds_list = origin_embeds_dict[k]
                    target_embeds = target_embeds_list[i]
                    origin_embeds = origin_embeds_list[i]
                    
                    if loss_choice == 'mse':
                        Loss_x = wi * F.mse_loss(adv_embeds, target_embeds, reduction="mean")
                        Loss_d = -F.mse_loss(adv_embeds, origin_embeds, reduction="mean")
                    else:  # cosine
                        Loss_x = -F.cosine_similarity(adv_embeds, target_embeds, -1).mean()
                        Loss_d = F.cosine_similarity(adv_embeds, origin_embeds, -1).mean()
                    
                    loss_x_list.append(Loss_x)
                    loss_d_list.append(Loss_d)
                
                # EOT变换下的平均损失
                mean_Loss_x = torch.stack(loss_x_list).mean()
                mean_Loss_d = torch.stack(loss_d_list).mean()
                Loss_x_.append(mean_Loss_x)
                Loss_d_.append(mean_Loss_d)
            
            # 组合损失
            Loss_x_ = torch.stack(Loss_x_).view(1, len(self.model_dict.keys()))
            Loss_d_ = torch.stack(Loss_d_).view(1, len(self.model_dict.keys()))
            Loss_ = (1 - w) * Loss_x_ + w * Loss_d_
            Loss_ = Loss_.mean()
            
            Loss_dict[epoch] = [Loss_.item(), Loss_x_.mean().item(), Loss_d_.mean().item()]
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Loss = {Loss_.item():.4f}")
            
            # 计算梯度并更新
            # print("计算梯度并更新扰动")
            grad = torch.autograd.grad(Loss_, perturbed_data)[0]
            adv_perturbed_data = perturbed_data - alpha * grad.sign()
            
            # 投影到eps球内
            # print("投影扰动到eps球内")
            et = torch.clamp(adv_perturbed_data - origin_data, min=-eps, max=+eps)
            perturbed_data = torch.clamp(origin_data + et, min=0, max=255).detach().clone()
        
        return perturbed_data.cpu(), Loss_dict
    
    def protect_image(self, input_image: Image.Image, parameters: Dict[str, Any]) -> Image.Image:
        """保护单张图像 - 使用集成的PGD攻击算法"""
        # try:
        # 转换输入图像为tensor
        # print("转换输入图像为tensor")
        input_array = np.array(input_image).astype(np.uint8)
        input_tensor = torch.from_numpy(input_array).permute(2, 0, 1).unsqueeze(0).float()
        
        # 设置模型和变换
        # print("设置模型和变换")
        precision = "fp32"
        if precision == "fp32":
            torch_dtype = torch.float32
        elif precision == "fp16":
            torch_dtype = torch.float16
        elif precision == "bf16":
            torch_dtype = torch.bfloat16
        else:
            raise ValueError("precision must be one of [fp32, fp16, bf16]")
        self.setup_models(parameters['model_types'], precision)
        self.setup_transforms(parameters['eot_trans_types'], parameters['resample_interpolation'])
        
        # 加载目标数据
        # print("加载目标数据")
        target_tensor = self.load_target_data(parameters['target_type'], parameters['input_size'])
        
        # 准备数据
        origin_data = input_tensor.detach().clone().to(self.device).requires_grad_(False).to(torch_dtype)
        perturbed_data = input_tensor.to(self.device).requires_grad_(True).to(torch_dtype)
        target_data = target_tensor.to(self.device).requires_grad_(False).to(torch_dtype)
        
        # 应用PGD集成攻击
        # print("应用PGD集成攻击")
        alpha_val = parameters['alpha'] * 255  # 缩放步长
        eps_val = parameters['eps']  # 噪声预算

        protected_tensor, loss_dict = self.pgd_ensemble_attack(
            perturbed_data=perturbed_data,
            origin_data=origin_data,
            alpha=alpha_val,
            eps=eps_val,
            attack_num=parameters['attack_num'],
            target_data=target_data,
            loss_choice=parameters['loss_choice'],
            w=parameters['w']
        )
        for epoch, losses in loss_dict.items():
            print(f"Final Epoch {epoch}: Total Loss = {losses[0]:.4f}, Target Loss = {losses[1]:.4f}, Deviation Loss = {losses[2]:.4f}")
        # 转换回PIL图像
        protected_array = protected_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        protected_image = Image.fromarray(protected_array)
        
        return protected_image
        
        # except Exception as e:
        #     print(f"Error in protect_image: {e}")
        #     # 出错时返回原始图像
        #     return input_image

def create_protector_interface():
    """创建Gradio界面"""
    
    protector = ImageProtector()
    
    def protect_single_image(input_image, 
                           model_types, 
                           attack_num, 
                           loss_choice,
                           w,
                           alpha,
                           eps,
                           input_size,
                           eot_trans_types,
                           resample_interpolation,
                           target_type):
        """保护单张图像"""
        
        # 设置参数
        parameters = {
            'model_types': model_types,
            'attack_num': attack_num,
            'loss_choice': loss_choice,
            'w': w,
            'alpha': alpha,
            'eps': eps,
            'input_size': input_size,
            'eot_trans_types': eot_trans_types,
            'resample_interpolation': resample_interpolation,
            'target_type': target_type
        }
        
        if input_image is None:
            return None, "请先上传图像"
        
        try:
            # 调整图像大小
            if input_size != input_image.size[0]:
                input_image = input_image.resize((input_size, input_size), Image.BILINEAR)
            
            # 保护图像
            protected_image = protector.protect_image(input_image, parameters)
            return protected_image, "图像保护完成！"
            
        except Exception as e:
            return None, f"处理图像时出错: {str(e)}"
    
    def protect_batch_images(input_folder, 
                           model_types, 
                           attack_num, 
                           loss_choice,
                           w,
                           alpha,
                           eps,
                           input_size,
                           eot_trans_types,
                           resample_interpolation,
                           target_type):
        """保护批量图像"""
        
        if input_folder is None:
            return None, "请先选择文件夹"
        # print(f"Processing folder: {input_folder}")
        try:
            # 处理文件夹中的所有图像
            output_dir = tempfile.mkdtemp()
            image_files = []
            
            if isinstance(input_folder, list):
                # Gradio返回文件列表，每个元素是临时文件对象
                for file_obj in input_folder:
                    if file_obj.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_files.append(file_obj.name)
            else:
                # 如果是字符串路径（备用）
                for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']:
                    image_files.extend(glob.glob(f"{input_folder}/**/{ext}", recursive=True))
            
            if not image_files:
                return None, "文件夹中没有找到图像文件"
            
            processed_images = []
            parameters = {
                'model_types': model_types,
                'attack_num': attack_num,
                'loss_choice': loss_choice,
                'w': w,
                'alpha': alpha,
                'eps': eps,
                'input_size': input_size,
                'eot_trans_types': eot_trans_types,
                'resample_interpolation': resample_interpolation,
                'target_type': target_type
            }
            
            for i, image_file in enumerate(image_files[:8]):  # 限制处理数量
                try:
                    img = Image.open(image_file)
                    if input_size != img.size[0]:
                        img = img.resize((input_size, input_size), Image.BILINEAR)
                    # print("开始保护图像")
                    protected_img = protector.protect_image(img, parameters)
                    processed_images.append(protected_img)
                    
                except Exception as e:
                    print(f"处理图像 {image_file} 时出错: {e}")
            
            if not processed_images:
                return None, "没有成功处理任何图像"
            
            return processed_images, f"成功处理 {len(processed_images)} 张图像"
            
        except Exception as e:
            return None, f"处理文件夹时出错: {str(e)}"
    
    # 创建界面
    with gr.Blocks(title="图像保护系统", theme="soft") as interface:
        gr.Markdown("# 🛡️ 图像保护系统")
        gr.Markdown("使用对抗性攻击技术保护您的图像免遭未经授权的AI模型使用")
        
        with gr.Tab("单张图像保护"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="上传待保护图像", type="pil")
                    
                    with gr.Accordion("攻击参数", open=False):
                        model_types = gr.CheckboxGroup(
                            choices=["clip", "ipadapter", "photomaker", "face_diffuser"],
                            value=["clip", "photomaker"],
                            label="目标模型类型"
                        )
                        
                        attack_num = gr.Slider(
                            minimum=10, maximum=500, value=100, step=10,
                            label="攻击迭代次数"
                        )
                        
                        loss_choice = gr.Radio(
                            choices=["cosine", "mse"], value="cosine",
                            label="损失函数类型"
                        )
                        
                        w = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                            label="权重参数 w"
                        )
                        
                        alpha = gr.Slider(
                            minimum=0.001, maximum=0.1, value=0.005, step=0.001,
                            label="步长 alpha"
                        )
                        
                        eps = gr.Slider(
                            minimum=1, maximum=32, value=16, step=1,
                            label="噪声预算 eps"
                        )
                    
                    with gr.Accordion("图像参数", open=False):
                        input_size = gr.Slider(
                            minimum=224, maximum=1024, value=512, step=32,
                            label="输入图像尺寸"
                        )
                        
                        eot_trans_types = gr.CheckboxGroup(
                            choices=["none", "gau", "hflip", "gau-hflip"],
                            value=["none"],
                            label="EOT变换类型"
                        )
                        
                        resample_interpolation = gr.Radio(
                            choices=["BILINEAR", "BICUBIC"], value="BILINEAR",
                            label="重采样插值方法"
                        )
                        
                        target_type = gr.Radio(
                            choices=["max", "yingbu", "mist", "colored_mist", "gray"],
                            value="max",
                            label="目标图像类型"
                        )
                    
                    protect_btn = gr.Button("🛡️ 保护图像", variant="primary")
                
                with gr.Column():
                    output_image = gr.Image(label="保护后的图像", interactive=False)
                    status_text = gr.Textbox(label="状态信息", interactive=False)
        
        with gr.Tab("批量图像保护"):
            with gr.Row():
                with gr.Column():
                    input_folder = gr.File(
                        label="选择图像文件夹",
                        file_count="directory"
                    )
                    
                    # 复用相同的参数
                    with gr.Accordion("攻击参数", open=False):
                        batch_model_types = gr.CheckboxGroup(
                            choices=["clip", "ipadapter", "photomaker", "face_diffuser"],
                            value=["clip", "photomaker"],
                            label="目标模型类型"
                        )
                        
                        batch_attack_num = gr.Slider(
                            minimum=10, maximum=500, value=100, step=10,
                            label="攻击迭代次数"
                        )
                        
                        batch_loss_choice = gr.Radio(
                            choices=["cosine", "mse"], value="cosine",
                            label="损失函数类型"
                        )
                        
                        batch_w = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                            label="权重参数 w"
                        )
                        
                        batch_alpha = gr.Slider(
                            minimum=0.001, maximum=0.1, value=0.005, step=0.001,
                            label="步长 alpha"
                        )
                        
                        batch_eps = gr.Slider(
                            minimum=1, maximum=32, value=16, step=1,
                            label="噪声预算 eps"
                        )
                    
                    with gr.Accordion("图像参数", open=False):
                        batch_input_size = gr.Slider(
                            minimum=224, maximum=1024, value=512, step=32,
                            label="输入图像尺寸"
                        )
                        
                        batch_eot_trans_types = gr.CheckboxGroup(
                            choices=["none", "gau", "hflip", "gau-hflip"],
                            value=["none"],
                            label="EOT变换类型"
                        )
                        
                        batch_resample_interpolation = gr.Radio(
                            choices=["BILINEAR", "BICUBIC"], value="BILINEAR",
                            label="重采样插值方法"
                        )
                        
                        batch_target_type = gr.Radio(
                            choices=["max", "yingbu", "mist", "colored_mist", "gray"],
                            value="max",
                            label="目标图像类型"
                        )
                    
                    batch_protect_btn = gr.Button("🛡️ 批量保护图像", variant="primary")
                
                with gr.Column():
                    batch_output_gallery = gr.Gallery(
                        label="保护后的图像",
                        show_label=True,
                        elem_id="gallery",
                        columns=2,
                        height="auto"
                    )
                    batch_status_text = gr.Textbox(label="状态信息", interactive=False)
        
        # 绑定事件
        protect_btn.click(
            fn=protect_single_image,
            inputs=[
                input_image, model_types, attack_num, loss_choice, w, alpha, eps,
                input_size, eot_trans_types, resample_interpolation, target_type
            ],
            outputs=[output_image, status_text]
        )
        
        batch_protect_btn.click(
            fn=protect_batch_images,
            inputs=[
                input_folder, batch_model_types, batch_attack_num, batch_loss_choice,
                batch_w, batch_alpha, batch_eps, batch_input_size, batch_eot_trans_types,
                batch_resample_interpolation, batch_target_type
            ],
            outputs=[batch_output_gallery, batch_status_text]
        )
        
        # 添加说明
        with gr.Accordion("使用说明", open=False):
            gr.Markdown("""
            ## 图像保护系统使用指南
            
            ### 功能说明
            - **单张图像保护**: 上传单张图像进行保护处理
            - **批量图像保护**: 上传包含多张图像的文件夹进行批量处理
            
            ### 核心算法
            本系统使用基于PGD(Projected Gradient Descent)的集成攻击算法，通过添加人眼难以察觉的对抗性噪声，
            使AI模型无法正确识别和处理受保护的图像。
            
            ### 参数说明
            - **目标模型类型**: 选择要防御的AI模型类型
            - **攻击迭代次数**: 对抗性攻击的迭代次数，值越大效果越好但耗时越长
            - **损失函数类型**: 选择用于优化的损失函数(余弦相似度或均方误差)
            - **权重参数 w**: 平衡目标损失和偏差损失的权重
            - **步长 alpha**: 每次攻击迭代的步长大小
            - **噪声预算 eps**: 允许添加的最大噪声量
            - **EOT变换类型**: 期望过度变换的类型，用于增强攻击的鲁棒性
            
            ### 使用建议
            1. 首先使用默认参数尝试单张图像保护
            2. 根据效果调整参数(建议优先调整eps和attack_num)
            3. 确认效果满意后进行批量处理
            """)
    
    return interface

if __name__ == "__main__":
    # 创建并启动Gradio界面
    interface = create_protector_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )