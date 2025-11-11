##### Table of contents
1. [Environment setup](#environment-setup)
2. [Dataset preparation](#dataset-preparation)
3. [How to run](#how-to-run)
4. [Contacts](#contacts)

# Official PyTorch implementation of "FaceOff: Preventing Unauthorized Text-to-Image Identity Customization"
<div align="center">
  <a href="https://orcid.org/0009-0004-4845-4244" target="_blank">Mingwang Hu</a> &emsp;
  <a href="" target="_blank">Chenyu Zhang</a> &emsp;
  <a href="" target="_blank">Zili Yi</a> &emsp;
  <a href="https://orcid.org/0000-0002-7696-5330" target="_blank">Lanjun Wang</a>
  <br> <br>
</div>
<br>


> **Abstract**: Recently, encoder-based methods such as PhotoMaker have advanced text-to-image identity (ID) customization. Unlike tuning-based approaches like DreamBooth, these methods employ powerful facial encoders to extract ID information from a singl portrait, enabling efficient customization in a single inference pass. However, their misuse can exacerbate the generation of misleading and harmful content, endangering individuals and society. To address this problem, existing methods introduce protective perturbations into user portrait images to distort ID information in the customized images. We identify two limitations of these methods: 1) copyright infringement risk caused by identified faces in customized images; 2) lack of robustness against adversarial purification. To address these issues, we propose FaceOff, a framework that protects portrait images from the threat of encoder-based ID customization by disturbing the ensemble of image encoders. Specifically, to reduce the risk of copyright infringement, we design a contrastive loss to shift the image semantics to a target ID and deviate the image semantics away from the original ID. In addition, we introduce a Gaussian augmentation module to mitigate the protection degradation under adversarial purification. FaceOff outperforms the SOTA method by 12.65% of FDFR and 4.70% of ISM across three customization models and two facial benchmarks on average.

**Keywords**:  DeepFake Defense, Adversarial Examples, Diffusion Model, Identity Customization

## Environment setup

Install dependencies:
```shell
cd FaceOff
conda create -n faceoff python=3.10  
conda activate faceoff  
pip install -r requirements.txt  
```

We also provide a conda environment file for easy setup:
```shell
conda env create -f environment.yaml
```

## Model preparation
We use three popular encoder-based ID customization models in our paper: [PhotoMaker](https://github.com/TencentARC/PhotoMaker), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) and [IP-Adapter Plus](https://github.com/tencent-ailab/IP-Adapter). We have provide modified implementations under the folder ```customization/target_model```. The base diffusion model is [SDXL base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

## Dataset preparation
We process the two datasets VGGFace2 and CelebA-HQ used in [Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth). You can download the processed datasets from our [Google Drive](https://drive.google.com/file/d/1VwgZBqoPybVtcjBv5u41vDQMGbyugLYD/view?usp=sharing).

## How to run
- attack
    - To attack IP-Adapter, run: 
    ```
    sh ./scripts/attack/faceoff.sh
    ```
    - To attack DreamBooth, run:
    ```
    sh ./scripts/attack/faceoff_dreambooth.sh
    ```
    - To attack Face-diffuser and Fastcomposer, run:
    ```shell
    sh ./scripts/attack/faceoff_face_diffuser.sh
    ```
- customization
    - To customize with IP-Adapter, run:
    ```shell
    sh ./scripts/gen/gen_ipadapter_sd1-5.sh
    ```
    - To customize with DreamBooth, use the same implementation as in [Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth)
    - To customize with Face-diffuser and Fastcomposer, use the same implementation as in [Face-diffuser](https://github.com/CodeGoat24/Face-diffuser) and [FastComposer](https://github.com/mit-han-lab/fastcomposer)
- evaluation
   - We follow [Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth) and [MetaCloak](https://github.com/liuyixin-louis/MetaCloak) to evaluate the customized images and protected images:
   ```
   sh ./scripts/eval/eval_new_ip-adapter.sh
   ```
- purification
    - We follow [MetaCloak](https://github.com/liuyixin-louis/MetaCloak) to purify the protected images with JPEG, SR or TVM:
    ```shell
    sh ./scripts/purif/purification.sh
    ```

## Acknowledgement
- [Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth)
- [MetaCloak](https://github.com/liuyixin-louis/MetaCloak)
- [PhotoMaker](https://github.com/TencentARC/PhotoMaker)
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
