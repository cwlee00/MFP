# [CVPR 2024] MFP
Chaewon Lee,
Seon-Ho Lee, 
and Chang-Su Kim

Official code for **"MFP: Making Full Use of Probability Maps for Interactive Image Segmentation"**[[paper]](https://arxiv.org/abs/2404.18448)

### Requirements
- PyTorch 1.11.0
- CUDA 11.3
- CuDNN 8.2.0
- python 3.8
  
### Installation
Create conda environment:
```bash
    $ conda create -n MFP python=3.8 anaconda
    $ conda activate MFP
    $ conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
    $ pip install -r requirements.txt
```
Download repository:
```bash
    $ git clone https://github.com/cwlee00/MFP.git
```
Download weights:

MFP model [Google Drive](https://drive.google.com/drive/folders/1ygeSwkVfGlydP-LW6YnhSCyed4kLOe0f?usp=sharing)

### Evaluation
For evaluation, please download the datasets and models, and then configure the path in [config.yml](https://github.com/cwlee00/MFP/blob/main/config.yml)

```
python scripts/evaluate_model.py NoBRS \
--gpu=0 \
--checkpoint=./weights/mfp_models/MFP_vit_base(cocolvis).pth \
--eval-mode=cvpr \
--datasets=GrabCut,Berkeley,DAVIS,SBD
```
### Train
For training, please download the [MAE](https://github.com/facebookresearch/mae) pretrained weights (click to download: [ViT-Base](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)) and configure the dowloaded path in [config.yml](https://github.com/cwlee00/MFP/blob/main/config.yml).

```
python train.py models/iter_mask/plainvit_base448_cocolvis_itermask_prevMod.py \
--batch-size=8 \
--ngpus=1
```

### Citation
Please cite the following paper if you feel this repository useful.
```bibtex
    @InProceedings{Lee_2024_CVPR,
    author    = {Lee, Chaewon and Lee, Seon-Ho and Kim, Chang-Su},
    title     = {MFP: Making Full Use of Probability Maps for Interactive Image Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {4051-4059}
    }
```
### Acknowledgement
Our project is developed based on [RITM](https://github.com/saic-vul/ritm_interactive_segmentation) and [SimpleClick](https://github.com/uncbiag/SimpleClick). We would like to show sincere thanks to the contributors. 
