"""Standard GAN (DCGAN-style) on MNIST, non-saturating loss.

G maps noise z -> 28x28 image, D classifies real vs. fake. They are trained alternately.
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
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--z-dim", type=int, default=64)
p.add_argument("--sample-every", type=int, default=5)
p.add_argument("--data", default=str(HERE.parent / "data"))
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0)

# images scaled to [-1, 1] to match the tanh output of the generator
data = datasets.MNIST(args.data, train=True, download=True).data
data = (data.float().div(255) * 2 - 1).unsqueeze(1).to(device)  # [60000, 1, 28, 28]

G = nn.Sequential(
    nn.Linear(args.z_dim, 128 * 7 * 7), nn.BatchNorm1d(128 * 7 * 7), nn.ReLU(),
    nn.Unflatten(1, (128, 7, 7)),
    nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),  # 7 -> 14
    nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh(),  # 14 -> 28
).to(device)
D = nn.Sequential(
    nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),  # 28 -> 14
    nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),  # 14 -> 7
    nn.Flatten(), nn.Linear(128 * 7 * 7, 1),  # one logit: "is real"
).to(device)

opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
bce = F.binary_cross_entropy_with_logits

out = HERE / "outputs"
out.mkdir(exist_ok=True)
fixed_z = torch.randn(64, args.z_dim, device=device)  # same noise every epoch -> see training progress
losses = []

for epoch in range(1, args.epochs + 1):
    perm = torch.randperm(len(data), device=device)
    for i in range(0, len(perm) - args.batch_size + 1, args.batch_size):
        real = data[perm[i:i + args.batch_size]]
        fake = G(torch.randn(len(real), args.z_dim, device=device))
        ones, zeros = torch.ones(len(real), 1, device=device), torch.zeros(len(real), 1, device=device)

        # discriminator step: real -> 1, fake -> 0
        loss_d = bce(D(real), ones) + bce(D(fake.detach()), zeros)
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # generator step: make D say "real" for fakes
        loss_g = bce(D(fake), ones)
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
        losses.append((loss_d.item(), loss_g.item()))

    print(f"epoch {epoch}/{args.epochs}  loss D {loss_d.item():.3f}  loss G {loss_g.item():.3f}", flush=True)
    if epoch % args.sample_every == 0 or epoch == args.epochs:
        G.eval()
        with torch.no_grad():
            save_image(G(fixed_z), out / f"samples_epoch_{epoch:03d}.png", nrow=8, normalize=True, value_range=(-1, 1))
        G.train()

plt.figure(figsize=(6, 3))
plt.plot([l[0] for l in losses], label="D", alpha=0.7)
plt.plot([l[1] for l in losses], label="G", alpha=0.7)
plt.xlabel("iteration"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
plt.savefig(out / "losses.png", dpi=120)
torch.save(G.state_dict(), out / "generator.pt")
print(f"results in {out}")
