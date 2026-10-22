"""Denoising diffusion probabilistic model (DDPM, Ho et al. 2020) on MNIST.

Forward process: x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps   (closed form, any t)
Training:        a small U-Net predicts eps from (x_t, t); loss = MSE(eps_pred, eps)
Sampling:        start from pure noise and apply the reverse step T times.
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--timesteps", type=int, default=1000)
p.add_argument("--data", default=str(HERE.parent / "data"))
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0)

data = datasets.MNIST(args.data, train=True, download=True).data
data = (data.float().div(255) * 2 - 1).unsqueeze(1).to(device)  # [60000, 1, 28, 28] in [-1, 1]

# ------------------------------------------------------------- noise schedule
T = args.timesteps
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1 - betas
abar = torch.cumprod(alphas, 0)  # abar_t = prod_{s<=t} alpha_s


def q_sample(x0, t, eps):
    """Forward process: jump directly from x_0 to x_t."""
    a = abar[t].view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1 - a).sqrt() * eps


# --------------------------------------------------------------------- model
def time_embedding(t, dim):
    freqs = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    a = t[:, None].float() * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]  # inject the timestep
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, ch=64, tdim=128):
        super().__init__()
        self.tdim = tdim
        self.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(1, ch, 3, padding=1)
        self.down1 = ResBlock(ch, ch, tdim)  # 28x28
        self.down2 = ResBlock(ch, 2 * ch, tdim)  # 14x14
        self.pool = nn.AvgPool2d(2)
        self.mid = ResBlock(2 * ch, 2 * ch, tdim)  # 7x7
        self.up2 = ResBlock(4 * ch, 2 * ch, tdim)  # 14x14, concat with skip
        self.up1 = ResBlock(3 * ch, ch, tdim)  # 28x28, concat with skip
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x, t):
        temb = self.temb(time_embedding(t, self.tdim))
        h1 = self.down1(self.inp(x), temb)  # 28
        h2 = self.down2(self.pool(h1), temb)  # 14
        h = self.mid(self.pool(h2), temb)  # 7
        h = self.up2(torch.cat([F.interpolate(h, scale_factor=2), h2], 1), temb)  # 14
        h = self.up1(torch.cat([F.interpolate(h, scale_factor=2), h1], 1), temb)  # 28
        return self.out(h)


model = UNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
print(f"U-Net with {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")

# ------------------------------------------------------------------ training
for epoch in range(1, args.epochs + 1):
    perm = torch.randperm(len(data), device=device)
    for i in range(0, len(perm) - args.batch_size + 1, args.batch_size):
        x0 = data[perm[i:i + args.batch_size]]
        t = torch.randint(0, T, (len(x0),), device=device)  # a random timestep per image
        eps = torch.randn_like(x0)
        loss = F.mse_loss(model(q_sample(x0, t, eps), t), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {epoch}/{args.epochs}  loss {loss.item():.4f}", flush=True)

# ------------------------------------------------------------------ sampling
out = HERE / "outputs"
out.mkdir(exist_ok=True)


@torch.no_grad()
def sample(n, snapshots=()):
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)
    snaps = []
    for t in reversed(range(T)):
        tt = torch.full((n,), t, device=device)
        eps = model(x, tt)
        mean = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()
        x = mean + betas[t].sqrt() * torch.randn_like(x) if t > 0 else mean
        if t in snapshots:
            snaps.append(x.clone())
    return x, snaps


# forward process: one image, increasingly noisy (t = 0, 100, ..., 900, 999)
ts = torch.tensor([0, 100, 200, 300, 400, 500, 700, 999], device=device)
x0 = data[:1].expand(len(ts), -1, -1, -1)
save_image(q_sample(x0, ts, torch.randn_like(x0)), out / "forward_process.png", nrow=len(ts), normalize=True, value_range=(-1, 1))

# generated samples, plus the reverse process for the first 8 of them (columns = t from T-1 down to 0)
snap_t = [T - 1] + [int(T * (1 - k / 8)) for k in range(1, 8)] + [0]
x, snaps = sample(64, snapshots=set(snap_t))
save_image(x, out / "samples.png", nrow=8, normalize=True, value_range=(-1, 1))
save_image(torch.stack([s[:8] for s in snaps], 1).flatten(0, 1), out / "reverse_process.png",
           nrow=len(snaps), normalize=True, value_range=(-1, 1))
torch.save(model.state_dict(), out / "ddpm_unet.pt")
print(f"results in {out}")
