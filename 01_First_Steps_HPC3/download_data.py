"""Download MNIST once (run this on the LOGIN node, which has internet access).

The compute nodes then simply read the files from the shared home directory.
"""
from pathlib import Path
from torchvision import datasets

root = Path(__file__).resolve().parent.parent / "data"
datasets.MNIST(root, train=True, download=True)
datasets.MNIST(root, train=False, download=True)
print(f"MNIST is in {root}")
