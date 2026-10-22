"""Variational autoencoder on MNIST (MLP encoder/decoder, Gaussian latent).

loss = reconstruction (Bernoulli NLL) + KL( q(z|x) || N(0, I) ).
With --latent-dim 2 the script also plots the latent space and a decoded manifold.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--latent-dim", type=int, default=2)
p.add_argument("--data", default=str(HERE.parent / "data"))
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0)


def load(train):
    ds = datasets.MNIST(args.data, train=train, download=True)
    return ds.data.float().div(255).flatten(1).to(device), ds.targets  # pixels in [0, 1]


train_x, _ = load(True)
test_x, test_y = load(False)


class VAE(nn.Module):
    def __init__(self, zdim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.mu, self.logvar = nn.Linear(256, zdim), nn.Linear(256, zdim)
        self.dec = nn.Sequential(nn.Linear(zdim, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 784))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)  # reparameterization trick
        return self.dec(z), mu, logvar


def loss_fn(logits, x, mu, logvar):
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())  # closed form for Gaussians
    return (rec + kl) / len(x), rec / len(x), kl / len(x)


model = VAE(args.latent_dim).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(1, args.epochs + 1):
    perm = torch.randperm(len(train_x), device=device)
    for i in range(0, len(perm) - args.batch_size + 1, args.batch_size):
        x = train_x[perm[i:i + args.batch_size]]
        logits, mu, logvar = model(x)
        loss, rec, kl = loss_fn(logits, x, mu, logvar)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {epoch}/{args.epochs}  loss {loss.item():.2f}  (rec {rec.item():.2f} + KL {kl.item():.2f})", flush=True)

out = HERE / "outputs"
out.mkdir(exist_ok=True)
model.eval()
with torch.no_grad():
    # reconstructions: top row = test images, bottom row = reconstructions (from the mean of q(z|x))
    x = test_x[:16]
    mu = model.mu(model.enc(x))
    recon = torch.sigmoid(model.dec(mu))
    save_image(torch.cat([x, recon]).view(-1, 1, 28, 28), out / "reconstructions.png", nrow=16)

    # generation: z ~ N(0, I) -> decoder
    z = torch.randn(64, args.latent_dim, device=device)
    save_image(torch.sigmoid(model.dec(z)).view(-1, 1, 28, 28), out / "samples.png", nrow=8)

    if args.latent_dim == 2:
        mus = model.mu(model.enc(test_x)).cpu()
        g = torch.linspace(-3, 3, 20)  # decode a regular grid in latent space
        grid = torch.stack(torch.meshgrid(g, g.flip(0), indexing="xy"), -1).view(-1, 2).to(device)
        imgs = torch.sigmoid(model.dec(grid)).view(20, 20, 28, 28).permute(0, 2, 1, 3).reshape(560, 560).cpu()
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sc = ax[0].scatter(mus[:, 0], mus[:, 1], c=test_y, cmap="tab10", s=3)
        ax[0].legend(*sc.legend_elements(), loc="upper right", fontsize=7)
        ax[0].set_title("encoded test set (mean of q(z|x))")
        ax[1].imshow(imgs, cmap="gray", extent=(-3, 3, -3, 3))
        ax[1].set_title("decoded latent grid")
        plt.tight_layout()
        plt.savefig(out / "latent_space.png", dpi=120)
torch.save(model.state_dict(), out / "vae.pt")
print(f"results in {out}")
