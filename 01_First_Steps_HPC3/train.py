"""Train a small CNN on MNIST. Meant as a 'hello world' for HPC3:
it prints where it runs (node, job id, GPU, VRAM share) and then trains."""
import argparse
import os
import socket
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------- model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 28 -> 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 14 -> 7
        return self.fc(x.flatten(1))


# DataLoader workers re-import this file in new processes (the default on recent Python
# versions), so everything that runs must be protected by `if __name__ == "__main__"`.
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data", default=str(HERE.parent / "data"))
    args = p.parse_args()

    # ------------------------------------------------------------ environment
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"host={socket.gethostname()}  slurm_job_id={os.environ.get('SLURM_JOB_ID', '-')}  device={device}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()  # free memory right now, total = whole physical card
        print(f"GPU: {torch.cuda.get_device_name(0)}, {free / 1024**3:.1f} GB free of {total / 1024**3:.1f} GB")

    # ------------------------------------------------------------------- data
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])
    train_set = datasets.MNIST(args.data, train=True, download=True, transform=tf)
    test_set = datasets.MNIST(args.data, train=False, download=True, transform=tf)
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))  # 0 = main process (laptop); on the cluster: one worker per CPU
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_set, 1000, num_workers=workers)

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --------------------------------------------------------------- training
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        print(f"epoch {epoch}/{args.epochs}  loss {loss.item():.4f}  "
              f"test acc {correct / len(test_set):.4f}  ({time.time() - t0:.1f}s)", flush=True)

    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out / "mnist_cnn.pt")
    print(f"saved {out / 'mnist_cnn.pt'}")


if __name__ == "__main__":
    main()
