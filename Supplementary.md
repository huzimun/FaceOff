# Supplementary

<figure>
  <img src="./figs/framework.png" alt="Framework of FaceOff" width="800" height="450" align="center" >
  <figcaption>Figure A. Framework of FaceOff.</figcaption>
</figure>

## Framework of FaceOff
Figure A shows the Framework of FaceOff.


<figure>
  <img src="./figs/Table-A.png" alt="Transferability of FaceOff on unseen models" width="800" align="center" >
  <figcaption>Table A. Transferability of FaceOff on unseen models. The base model of IP-Adapter and IP-Adapter Plus has been replaced with SD v1.5. Fastcomposer and Face-diffuser are two unseen customization methods with the base model SD v1.5. The evaluation dataset is VGGFace2.</figcaption>
</figure>

## Broader models
In Table A, we evaluate FaceOff on unseen base models (SD v1.5) and customization methods (Fastcomposer and Face-diffuser). FaceOff raises the average FDFR from 0.020 to 0.167 and reduces the average ISM from 0.370 to 0.084, indicating its transferability on broader model coverage.

<figure>
  <img src="./figs/Table-B.png" alt=" Visual utility for protected images on VGGFace2" width="400" align="center" >
  <figcaption>Table B. Visual utility for protected images on VGGFace2. bold: best; underline: second best.</figcaption>
</figure>

## Visual utility
In Table B, we evaluate the visual utility of protected images. FaceOff obtains the best protection effectiveness while obtaining suboptimal visual utility. Further research is needed to explore the balance between the perturbation imperceptibility and protection effectiveness.

<figure>
  <img src="./figs/target_images.png" alt="Different target images evaluated in our paper" width="1000" align="center" >
  <figcaption>Figure B. Different target images evaluated in our paper.</figcaption>
</figure>

## Target images
In Figure B, we show the different target images evaluated in our paper, including gray, mist, noise, face, cartoon, and mask.
We evaluate the efficacy of FaceOff under different target images in Table C. 
<!-- TODO: add analysis -->
This is because encoder-based models are biased toward facial generation, while cartoon and mask images, despite having similar shapes, deviate from real facial semantics.

## Augmentation strategies
In the supplementary, we explore the impact of different augmentation strategies. We consider evaluation under Standard Customization and JPEG compression. All augmentation strategies can raise FDFR and reduce ISM values, indicating the flexibility of FaceOff.

<figure>
  <img src="./figs/ablation.png" alt="Visualization results of ablation study" width="800" align="center" >
  <figcaption>Figure D. Visualization results of ablation study.</figcaption>
</figure>

## More qualitative analysis
We supplement more visualization results of the robustness evaluation, ablation study, and target image analysis. 
In Figure C, under standard customization, the customized images of contrastive loss have the least recognized human faces, and the similarity between the recognized face and the original face is the lowest. 
While under three image transformations, the customized images of the Gaussian augmentation loss have the least recognized human faces, and the similarity between the recognized face and the original face is the lowest. For overall FaceOff, it exhibits balanced performance between standard transformation and three image transformations. 
In Figure D, the customized images of contrastive loss contain fewer human faces and have lower similarity with the original images compared to those with pure deviation loss. After combining with the Gaussian augmentation loss, the recognized human faces rate and face similarity in the customized further reduce. 
In Figure E, we provide the target images evaluated in this work. Fig.~\ref{} exhibits the customized images under different target images. The customized images under mask (Beijing opera of the role 'yingbu' ) and cartoon images have the lowest rate of recognized human faces and face similarity.