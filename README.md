# Dr.3D: Adapting 3D GANs to Artistic Drawings<br><sub>Official PyTorch Implementation of the SIGGRAPH ASIA 2022 Paper (Conference Track)</sub>

![Teaser image 1](./docs/image.png)

**Dr.3D: Adapting 3D GANs to Artistic Drawings**<br>
Wonjoon Jin, Nuri Ryu, Geonung Kim, Seung-Hwan Baek, Sunghyun Cho<br>

[\[Paper\]](https://jinwonjoon.github.io/dr3d/docs/assets/Dr3D/Dr3D_main.pdf)
[\[Supple\]](https://jinwonjoon.github.io/dr3d/docs/assets/Dr3D/Dr3D_supple.pdf)
[\[Project Page\]](https://jinwonjoon.github.io/dr3d/)

Abstract: *While 3D GANs have recently demonstrated the high-quality synthesis of multi-view consistent images and 3D shapes, they are mainly restricted to photo-realistic human portraits. This paper aims to extend 3D GANs to a different, but meaningful visual form: artistic portrait drawings. However, extending existing 3D GANs to drawings is challenging due to the inevitable geometric ambiguity present in drawings. To tackle this, we present Dr.3D, a novel adaptation approach that adapts an existing 3D GAN to artistic drawings. Dr.3D is equipped with three novel components to handle the geometric ambiguity: a deformation-aware 3D synthesis network, an alternating adaptation of pose estimation and image synthesis, and geometric priors. Experiments show that our approach can successfully adapt 3D GANs to drawings and enable multi-view consistent semantic editing of drawings.*


## Requirements
* We had experiments using V100, A100, A6000 and RTX3090 GPUs on Linux (Ubuntu 20.04).
* Python 3.8.12, Pytorch 1.11.0.
* For other python libraries, we present environment.yaml to create conda environment:
  - `conda env create -f environment.yaml`
  - `conda activate dr3d` or `source activate dr3d`


## Pretrained Models
Dr.3D needs checkpoints of (1) EG3D model pre-trained on FFHQ and (2) pose-estimation network pre-trained on FFHQ (Here, for more accurate pose estimation performance, we provide a checkpoint of a newly trained model using Hopenet https://github.com/natanielruiz/deep-head-pose instead of the pre-trained ResNet-based pose-estimation model as mentioned in the implementation detail in the original paper).

Therefore, you should install gdown to use the script below.
```bash
pip install gdown
pip install --upgrade gdown
```

After installing gdown, run the command below for the pre-trained checkpoints mentioned above.
```bash
bash downloads/download_ckpts.sh
```

If you want to get pretrained Dr.3D checkpoints, also run the command below.
```bash
bash downloads/download_dr3d.sh
```

If you fail to download the file using the bash scripts.
Then use this [link](https://drive.google.com/drive/folders/1RYsVu04DfRVUn88x397hjdlG4ww-h4FR?usp=sharing) to download checkpoints using google drive.

### Additional domains
We additionally train Dr.3D on [Naver-Webtoon dataset](https://github.com/bryandlee/naver-webtoon-data) dataset for showing diverse styles.

## Inference
To generate images or shapes using pre-trained checkpoints, download pretrained checkpoints then run the command below.
```bash
bash scripts/inference/gen_samples.sh
```

## Training
Before training Dr.3D, check configurations and datasets.
We use wandb for logging and presenting result images. 
* Please make sure to login wandb in your workspace.

# Configurations
Before training, check configurations of
* data: Dataset_path in your workspace.
* dataset_name: Dataset name (anything you want).
* outdir: Directory for saving logs and checkpoints.
* resume: Path for the pretrained EG3D checkpoint (default: ./ckpts/EG3D_FFHQ_ckpt.pth)
* posenet_path: Path for the pretrained pose-estimation network checkpoint (default: ./ckpts/hopenet_ckpt.pth)

# Dataset Tree
Dr3D does not need any labeled poses. Place images in the dataset on the dataset path ('data' in the configuration file.)
Structures of the files are as follows (Dr.3D can handle both structures),

```bash
../datasets/images/
├── img00000000.png
├── img00000001.png
├── img00000002.png
```

or

```bash
../datasets/images/
├── 00000
│   ├── img00000000.png
│   ├── img00000001.png
│   ├── img00000002.png
├── 00001
│   ├── img00000000.png
│   ├── img00000001.png
│   ├── img00000002.png
```


After preparation, run the code below to train Dr.3D.

```bash
python train.py --cfg_path ./configs/dr3d/your_own_config.yaml
```
For instance,
```bash
python train.py --cfg_path ./configs/dr3d/metface.yaml
```


## Real Image Reconstruction (Inversion)
To reconstruct a real target image, we provide codes for GAN inversion using the codes of Pivotal Tuning Inversion (https://github.com/danielroich/PTI).
```bash
python invert.py --cfg_path ./configs/inversion/your_own_config.yaml
```


## Acknowledgement
* We thank [Kyungmin Jo](https://scholar.google.com/citations?user=zyFvIS8AAAAJ&hl=ko&oi=ao) for refactoring EG3D codes and building configurations.
* Thanks for publicly sharing datasets of Anime ([StyleAnime](https://github.com/zsl2018/StyleAnime)) and Webtoon ([Naver-Webtoon dataset](https://github.com/zsl2018/StyleAnime)) styles.


# References
1. [Efficient Geometry-aware 3D Generative Adversarial Networks (EG3D)](https://arxiv.org/abs/2112.07945), Chan et al. 2022
2. [Pivotal Tuning for Latent-based Editing of Real Images](https://arxiv.org/abs/2106.05744), Roich et al. 2022 
3. [Fine-Grained Head Pose Estimation Without Keypoints](https://arxiv.org/abs/1710.00925), Ruiz et al. 2018

### Citation

```
@inproceedings{jin2022dr,
  title     = {Dr.3D: Adapting 3D GANs to Artistic Drawings},
  author    = {Jin, Wonjoon and Ryu, Nuri and Kim, Geonung and Baek, Seung-Hwan and Cho, Sunghyun},
  booktitle = {SIGGRAPH Asia 2022 Conference Papers},
  pages     = {1--8},
  year      = {2022}
}

```
