##### Table of contents
1. [Environment setup](#environment-setup)
2. [Model preparation](#model-preparation)
3. [Dataset preparation](#dataset-preparation)
4. [How to run](#how-to-run)
5. [Acknowledgement](#acknowledgement)
6. [<span style="color:red">Supplementary</span>](./Supplementary.md)
7. [TODO]: upload the revised paper

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

## Environment setup<a id="environment-setup"></a>
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

## Model preparation<a id="model-preparation"></a>
We use three popular encoder-based ID customization models in our paper: [PhotoMaker](https://github.com/TencentARC/PhotoMaker), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) and [IP-Adapter Plus](https://github.com/tencent-ailab/IP-Adapter). We have provided modified implementations under the folder ```customization/target_model```. The base diffusion model is [SDXL base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

## Dataset preparation<a id="dataset-preparation"></a>
We process the two datasets VGGFace2 and CelebA-HQ used in [Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth). You can download the processed datasets from our [Google Drive](https://drive.google.com/file/d/1VwgZBqoPybVtcjBv5u41vDQMGbyugLYD/view?usp=sharing).

## How to run<a id="how-to-run"></a>

- To attack, run:
  ```shell
  sh ./scripts/attack/ensemble_faceoff.sh
  ```
- To customization, run:
  ```shell
  sh ./scripts/gen/gen_ipadapter_ipadapter-plus_photomaker.sh
  ```
- To evaluation, run:
   ```shell
   sh ./scripts/eval/eval_photomaker_ipadapter_ipadapter-plus.sh
   ```

- purification
    - We follow [MetaCloak](https://github.com/liuyixin-louis/MetaCloak) to purify the protected images with JPEG, SR or TVM:
      ```shell
      sh ./scripts/purif/purification.sh
      ```
    - In our paper, we also evaluate FaceOff under two SOTA purification methods, [GrIDPure](https://github.com/ZhengyueZhao/GrIDPure) and [Robust-Style-Mimicry](https://github.com/ethz-spylab/robust-style-mimicry).

## Acknowledgement<a id="acknowledgement"></a>
- [Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth)
- [MetaCloak](https://github.com/liuyixin-louis/MetaCloak)
- [PhotoMaker](https://github.com/TencentARC/PhotoMaker)
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
