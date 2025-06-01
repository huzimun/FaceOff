# FaceOff: Preventing Unauthorized Text-to-Image Identity Customization

This repository contains the official implementation of FaceOff, a novel defense mechanism against unauthorized text-to-image identity customization.

## Rebuttal for AC
### Ablation study on the scaling factor $\lambda$
We conduct an ablation study on the the scaling factor $\lambda$ to analyze its impact on protection efficacy across four target models. 
To better compare with the individual target loss and deviation loss, we use a ratio $w$ to replace $\lambda$ in the FaceOff loss function:
- FaceOff loss function in our paper:

    $L_{FaceOff} = L_{target} + \lambda * L_{deviation}$

- Replace the scaling factor $\lambda$ with a ratio $w$:

    $L_{FaceOff} = (1-w) * L_{target} + w * L_{deviation}$

The ratio $w$ is set to 0, 0.25, 0.5, 0.75, and 1 respectively and the results are summarized in the following table. We can see that the best performance on IP-Adapter, Fastcomposer and Face-diffuser is achieved when $w=0.5$. For DreamBooth, the best performance is achieved when $w=0.25$.
We can also observe that the IMS values of w= 0.25, 0.5, and 0.75 are always less than the maximum IMS values of w=0 (i.e. target loss) and w=1 (i.e. deviation loss). That is to say, the joint FaceOff loss will not result in a worse situation than the individual target or deviation loss. This observation contributes to more diverse and effective perturbation patterns.

| Model  | IP-Adapter | IP-Adapter | Fastcomposer | Fastcomposer | Face-diffuser | Face-diffuser | DreamBooth | DreamBooth |
|--------|--------------------|--------------------|-----------------------|-----------------------|------------------------|------------------------|----------------------|----------------------|
| Method | $IMS_{ARC}$↓                      | $IMS_{VGG}$↓                      | $IMS_{ARC}$↓                        | $IMS_{VGG}$↓                        | $IMS_{ARC}$↓                        | $IMS_{VGG}$↓                        | $IMS_{ARC}$↓                        | $IMS_{VGG}$↓                        |
| No Def.| 0.37±0.08                    | 0.76±0.07                    | 0.38±0.079                     | 0.77±0.07                      | 0.39±0.08                      | 0.77±0.07                      | 0.57±0.06                      | 0.82±0.06                      |
| 0      | 0.05±0.06                    | 0.35±0.23                    | 0.18±0.09                      | 0.60±0.11                      | 0.18±0.08                      | 0.60±0.11                      | 0.16±0.10                      | 0.51±0.23                      |
| 0.25   | **0.04±0.04**                    | 0.32±0.22                    | 0.17±0.08                      | 0.58±0.12                      | 0.17±0.08                      | 0.59±0.12                      | **0.11±0.09**                      | **0.41±0.22**                      |
| 0.5    | **0.04±0.06**                  | **0.28±0.24**                    | **0.15±0.08**                      | **0.56±0.13**                      | **0.15±0.08**                      | **0.57±0.13**                     | 0.25±0.10                      | 0.64±0.18                      |
| 0.75 | 0.06±0.05                    | 0.43±0.13                    | **0.15±0.09**                      | 0.59±0.12                      | **0.15±0.09**                      | 0.60±0.13                      | 0.38±0.09                      | 0.75±0.10                      |
| 1      | 0.07±0.05           | 0.43±0.10           | 0.17±0.10             | 0.61±0.13             | 0.18±0.10              | 0.61±0.13              | 0.38±0.09            | 0.75±0.08            |


## Software Dependencies
We provide a conda environment file for easy setup. To create the environment, run:
```
conda env create -f environment.yaml
```

- faceoff_v2
    - 123

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