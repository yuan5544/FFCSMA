# yuan5544-Image-Inpainting-Algorithm-Based-on-Fast-Fourier-Convolution-and-Spatial-Mask-Attention
## Environment setup
- Python 3.7
- PyTorch >= 1.0 (test on PyTorch 1.0, 1.7.0)

conda create -n misf_env python=3.7

conda activate misf_env

conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch

pip install -r requirements.txt

## Train

python train.py
<br>
For the parameters: checkpoints/config.yml

## Test

Such as test on the places2 dataset, please following:
python test_one.py --img_path='./data/image/10.jpg' --mask_path='./data/mask/10_mask.png' --model_path='./checkpoints/places2_InpaintingModel_gen.pth'
