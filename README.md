# FaceOff: Preventing Unauthorized Text-to-Image Identity Customization

## Software Dependencies
- first, construct virtual environments for PhotoMaker follow https://github.com/TencentARC/PhotoMaker.
- second, install cv2 to support noise budget refiner as follows:
    ```
    pip install opencv-python
    ```

## Datasets and Pretrains
- Datasets: 
    - VGGFace2: https://github.com/ox-vgg/vgg_face2
- Pretrains
    - RealVisXL_v3.0: https://huggingface.co/SG161222/RealVisXL_V3.0
    - PhotoMaker: https://huggingface.co/TencentARC/PhotoMaker
## Scripts
- attack
    ```
    sh ./scripts/attack/faceoff.sh
    ```
- customization
    ```
    sh ./scripts/gen/photomaker.sh
    ```