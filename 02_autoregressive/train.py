"""Autoregressive image model: MNIST downscaled to 12x12 and binarized.

The image is a sequence of 144 binary pixels (row by row). An LSTM models
p(x) = prod_i p(x_i | x_<i). Sampling draws one pixel at a time from a Bernoulli.
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

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--data", default=str(HERE.parent / "data"))
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0)
S = 12  # image side length; sequence length is S*S


def load(train):
    x = datasets.MNIST(args.data, train=train, download=True).data.float().div(255).unsqueeze(1)
    x = F.interpolate(x, size=(S, S), mode="bilinear", antialias=True)
    return (x > 0.5).float().flatten(1).to(device)  # [N, 144], values in {0, 1}


train_x, test_x = load(True), load(False)


class PixelLSTM(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, state=None):  # x: [B, T, 1] -> logits [B, T]
        h, state = self.lstm(x, state)
        return self.fc(h).squeeze(-1), state


model = PixelLSTM().to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(1, args.epochs + 1):
    perm = torch.randperm(len(train_x), device=device)
    for i in range(0, len(perm) - args.batch_size + 1, args.batch_size):
        x = train_x[perm[i:i + args.batch_size]]  # targets: all 144 pixels
        inp = F.pad(x[:, :-1], (1, 0)).unsqueeze(-1)  # inputs: shifted right, a 0 stands in for "start"
        logits, _ = model(inp)
        loss = F.binary_cross_entropy_with_logits(logits, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        xt = test_x[:2000]
        logits, _ = model(F.pad(xt[:, :-1], (1, 0)).unsqueeze(-1))
        nll = F.binary_cross_entropy_with_logits(logits, xt, reduction="none").sum(1).mean()
    print(f"epoch {epoch}/{args.epochs}  train loss {loss.item():.4f}  test NLL {nll.item():.2f} nats/image", flush=True)


@torch.no_grad()
def generate(n, prefix=None):
    """Sample n images pixel by pixel. If prefix [n, k] is given, those k pixels are fixed (inpainting)."""
    model.eval()
    x = torch.zeros(n, S * S, device=device)
    inp, state = torch.zeros(n, 1, 1, device=device), None
    for i in range(S * S):
        logits, state = model(inp, state)  # the LSTM state carries x_<i, so one step per pixel suffices
        pix = torch.bernoulli(torch.sigmoid(logits[:, -1]))
        if prefix is not None and i < prefix.shape[1]:
            pix = prefix[:, i]
        x[:, i] = pix
        inp = pix.view(n, 1, 1)
    return x.view(n, S, S).cpu()


out = HERE / "outputs"
out.mkdir(exist_ok=True)

# 1) unconditional samples
imgs = generate(64)
fig, axes = plt.subplots(8, 8, figsize=(6, 6))
for ax, im in zip(axes.flat, imgs):
    ax.imshow(im, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
plt.tight_layout()
plt.savefig(out / "samples.png", dpi=120)

# 2) completion: keep the first k rows of a test image, let the model invent the rest
rows = [0, 3, 6, 9]
fig, axes = plt.subplots(6, 5, figsize=(7, 8))
for r in range(6):
    ref = test_x[r]
    axes[r, 0].imshow(ref.view(S, S).cpu(), cmap="gray")
    axes[r, 0].set_title("original" if r == 0 else "")
    for c, k in enumerate(rows, start=1):
        im = generate(1, ref[None, :k * S])[0]
        axes[r, c].imshow(im, cmap="gray", vmin=0, vmax=1)
        axes[r, c].axhline(k - 0.5, color="r", lw=2)
        axes[r, c].set_title(f"given {k} rows" if r == 0 else "")
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.savefig(out / "completions.png", dpi=120)
torch.save(model.state_dict(), out / "pixel_lstm.pt")
print(f"results in {out}")
