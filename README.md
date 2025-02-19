# FaceOff: Preventing Unauthorized Text-to-Image Identity Customization

## Software Dependencies
We provide a conda environment file for easy setup. To create the environment, run:
```
conda env create -f environment.yaml
```
## Datasets and Pretrains
- Datasets: 
    - VGGFace2: we use the preprocessed dataset from [Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth)
- Pretrains
    - [h94/IP-Adapter](https://hf-mirror.com/h94/IP-Adapter)
    - [Stable Diffusion v1-5](https://hf-mirror.com/stable-diffusion-v1-5/stable-diffusion-v1-5)
    - [openai/clip-vit-large-patch14](https://hf-mirror.com/openai/clip-vit-large-patch14)

## Scripts
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
    ```
    sh ./scripts/attack/faceoff_face_diffuser.sh
    ```
- customization
    - To customize with IP-Adapter, run:
    ```
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
    ```
    sh ./scripts/purif/purification.sh
    ```
## Acknowledgement
- [Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth)
- [MetaCloak](https://github.com/liuyixin-louis/MetaCloak)
- [PhotoMaker](https://github.com/TencentARC/PhotoMaker)
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)