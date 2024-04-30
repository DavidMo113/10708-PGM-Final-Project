# 10708-PGM-Final-Project

## Environment

```bash
conda create -n pgm python=3.9
conda activate pgm
# Our GPU supports CUDA 11.8
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Dataset

We use torchvision to load our data. Please directly refer to the training section.

## Training

Train different generative models with variant noise levels.

```python
python train_vae.py --noise_level <noise_level>
python train_gan.py --noise_level <noise_level>
python train_dm.py --noise_level <noise_level>
```

## Sampling

First put the results from different noise levels like the following. The number behind **res** is the noise level. We consider noise levels of 0, 0.1, 0.3, and 0.5.

```bash
res_0
├── dm
├── gan
└── vae

res_0.1
├── dm
├── gan
└── vae

res_0.3
├── dm
├── gan
└── vae

res_0.5
├── dm
├── gan
└── vae
```

Then run the following command to do the sampling processes.

```python
python sample.py
```

## Evaluation

```bash
bash run_fid.sh
```
