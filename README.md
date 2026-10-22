# Generative Artificial Intelligence – code examples

Minimal PyTorch examples for the lecture *Generative Artificial Intelligence* (HS Flensburg).
Every example is one self-contained `train.py` (~100 lines) that trains on MNIST, prints its progress
and writes figures to `outputs/`. They run on a laptop (CUDA, Apple MPS or CPU – picked automatically)
and on the HPC3 cluster. Each folder has a **README that walks through the code block by block** and shows
example outputs of the training progress.

| Folder | Model | Output |
|---|---|---|
| [`01_First_Steps_HPC3/`](01_First_Steps_HPC3/README.md) | CNN classifier, submitted to HPC3 (MPS GPU share) | log file, `mnist_cnn.pt` |
| [`02_autoregressive/`](02_autoregressive/README.md) | LSTM over the 144 pixels of binarized 12×12 MNIST | `samples.png`, `completions.png` |
| [`03_gan/`](03_gan/README.md) | DCGAN-style GAN (with architecture diagram) | `samples_epoch_*.png`, `losses.png` |
| [`04_vae/`](04_vae/README.md) | VAE (with architecture diagram; `--latent-dim 2` → latent-space plots) | `reconstructions.png`, `samples.png`, `latent_space.png` |
| [`05_ddpm/`](05_ddpm/README.md) | DDPM with a small U-Net | `forward_process.png`, `samples.png`, `reverse_process.png` |

## Quick start (local)

```bash
pip install -r requirements.txt
python 04_vae/train.py                 # MNIST is downloaded to ./data on first use
python 03_gan/train.py --epochs 10     # all scripts have --epochs, --batch-size, --lr, --data
```

## Running on HPC3

Start with [`01_First_Steps_HPC3/`](01_First_Steps_HPC3/README.md). To run any other example on the cluster,
copy `01_First_Steps_HPC3/job.sh`, change the last line (e.g. `srun python ../05_ddpm/train.py`) and
raise `--time` if needed (DDPM is the slowest example).
