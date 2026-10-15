# SSADRNet


This repository contains the **training and testing code** for **SSADRNet**, a medical
image segmentation framework. It supports two datasets:

- **Synapse** — multi-organ segmentation (9 classes).
- **ACDC** — cardiac segmentation (4 classes: background, RV, Myo, LV).

## Usage

### 1. Recommended environment

- Python 3.11
- PyTorch 2.8.0 + cu128 / torchvision 0.23.0 + cu128 (CUDA 12.8)

```bash
# create and activate a Python 3.11 environment
conda create -n ssadrnet python=3.11 -y
conda activate ssadrnet


# install PyTorch from the official cu128 index, then the remaining deps
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 2. Data preparation

**Synapse multi-organ dataset**

Sign up on the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480)
to download the dataset, or use the [preprocessed data](https://drive.google.com/file/d/1wvmw8DVyDKr5sOAFn5zUpfhbK4Vxjze4/view).
Place the data as:

```text
data/synapse/train_npz_new
data/synapse/test_vol_h5_new
lists/lists_Synapse
```

**ACDC dataset**

Download the [preprocessed ACDC dataset](https://drive.google.com/file/d/1CruCQ-jjvA97BX-LIYwXaRMLmp3DN9zc/view)
and place it as:

```text
data/ACDC/lists_ACDC
data/ACDC/test
data/ACDC/train
data/ACDC/valid
```

### 3. Pretrained backbone (MaxViT, ImageNet-256)

Download the pretrained MaxViT weights from
[Google Drive](https://drive.google.com/drive/folders/1wuJ8zekQpPfmydqVhfO1LYSKxrik4Ukf?usp=share_link)
and place the file `maxxvit_rmlp_small_rw_256_sw-37e217ff.pth` at:

```text
pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth
```

You can also point to a custom location via the environment variable:

```bash
export MERIT_PRETRAINED=/path/to/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth
```

### 4. Training

**Synapse**

```bash
python train_synapse.py
```

**ACDC**

```bash
python train_acdc.py
```



### 5. Testing

**Synapse**

```bash
python test_synapse.py --checkpoint 'your path'
```

**ACDC**

```bash
python test_acdc.py --checkpoint 'your path'
```



## Acknowledgement

We are very grateful for these excellent works
[timm](https://github.com/huggingface/pytorch-image-models),
[CASCADE](https://github.com/SLDGroup/CASCADE),
[PraNet](https://github.com/DengPingFan/PraNet),
[MERIT](https://github.com/SLDGroup/MERIT),
[Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) and
[TransUNet](https://github.com/Beckschen/TransUNet), which have provided the basis for
our framework.


