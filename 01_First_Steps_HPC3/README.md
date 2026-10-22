# First steps on HPC3

Goal: log in to the cluster, submit a job that trains an MNIST classifier on a (shared) GPU, watch it run,
and fetch the results. Replace `<kennung>` with your Hochschul-Kennung.

| File | Purpose |
|---|---|
| `train.py` | the actual program: small CNN on MNIST; prints node, job id, GPU and VRAM share |
| `setup_env.sh` | one-time: Python venv in `~/venvs/genai` |
| `job.sh` | the SLURM batch script (`sbatch job.sh`) |

## 0. Concepts in 60 seconds

- You log in to the **login node** via SSH. It is for editing files and submitting jobs – **never train there**.
- Work is described in a **batch script** and submitted with `sbatch`. SLURM puts it in a queue and starts it on a
  **compute node** as soon as the requested resources are free.
- **Partitions** are queues with different priority/time limits (`teaching`: 12 h, `interactive`: 4 h, `research`: 12 h).
- The GPUs come in three flavours: a full card (`--gres=gpu:full:1`), MIG slices (`--gres=gpu:mig_24gb:1`) and a
  **shared card via NVIDIA MPS** (`--gres=shard:1`, node `hpc3-perlman`). With MPS, several jobs run on the same
  physical GPU: VRAM is split into equal fixed shares, compute time is shared dynamically. That is what we use here –
  fine for small models like ours, but speed fluctuates with what the other jobs do.

## 1. Log in

You need to be in the campus network (or VPN) and have your public SSH key registered in the portal.

```bash
ssh <kennung>@hpc3.hs-flensburg.de
```

## 2. Get the code onto the cluster

The login node has internet access, so simply clone the repository **in the cluster shell**:

```bash
git clone https://github.com/VisuComp/GenAI.git
cd GenAI && git pull     # later: fetch updates
```

(If you have modified the code on your own computer and want to try it on the cluster without committing, copy it from your
computer instead: `rsync -avz --exclude outputs --exclude data ./ <kennung>@hpc3.hs-flensburg.de:~/GenAI/`.)

## 3. One-time setup (on the login node)

```bash
cd ~/GenAI/01_First_Steps_HPC3
bash setup_env.sh        # creates ~/venvs/genai, installs PyTorch (several GB!)
mkdir -p logs            # SLURM does NOT create the log directory; without it the job fails silently
```

MNIST needs no preparation: `train.py` downloads it automatically on first use (the compute nodes have internet access), into `../data`.

## 4. Submit the job

```bash
sbatch job.sh            # prints: Submitted batch job 12345
```

Read `job.sh` – every `#SBATCH` line is a resource request. The one that selects the MPS share is
`#SBATCH --gres=shard:1`.

## 5. Watch it

```bash
squeue --me                          # your jobs: PD = pending, R = running
tail -f logs/mnist-mps_12345.out     # live output (Ctrl-C stops tail, not the job)
scontrol show job 12345              # details, e.g. why it is still pending
sacct -j 12345 --format=JobID,State,Elapsed,MaxRSS    # accounting, also after the job has finished
scancel 12345                        # cancel the job
```

The log should look like this:

```
host=hpc3-perlman  slurm_job_id=7359  device=cuda
GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 93.5 GB free of 95.1 GB
epoch 1/5  loss ...  test acc ...  (...s)
...
saved .../01_First_Steps_HPC3/outputs/mnist_cnn.pt
```

Note the two numbers in the GPU line: `total` is the **whole physical card**, `free` is the memory that is still free on it
right now. `nvidia-smi` (also at the top of the log) always shows the whole card, plus one `nvidia-cuda-mps-server` process
per job that currently uses the shared GPU.

The cluster FAQ says that on an MPS share the free memory of a fresh job equals the share of your job (a fraction of the card).
In the run above it reported almost the whole card (93.5 of 95.1 GB), so do not assume that a limit is visible or enforced:
keep your memory use modest anyway, the GPU is shared with other people.

## 6. Fetch the results

From **your own computer**:

```bash
rsync -avz <kennung>@hpc3.hs-flensburg.de:~/GenAI/01_First_Steps_HPC3/outputs/ ./outputs/
```

## Things to try

1. Change `--gres=shard:1` to `--gres=gpu:mig_48gb:1` and compare epoch times in the log.
2. Request too little time (`--time=00:00:30`) and look at the final job state in `sacct`.
3. Submit the same job three times at once and watch `squeue --me` – do they share the GPU?
4. Run an interactive shell on a compute node: `srun --partition=interactive --gres=shard:1 --pty bash`.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Job ends immediately, no log file | `logs/` directory did not exist when you submitted |
| `ModuleNotFoundError: torch` in the log | venv not activated in `job.sh`, or `setup_env.sh` not run |
| `CUDA out of memory` although `nvidia-smi` shows free memory | you may have exceeded your MPS VRAM share; compare with the `free` value in the log |
| Job stays `PD` | nodes/shares busy; `scontrol show job <id>` shows the reason |
| Quota exceeded | PyTorch venvs are large – delete unused ones, check the "Speicher" page in the portal |

---

# Code walkthrough

## `job.sh` – the batch script

```bash
#!/bin/bash
#SBATCH --job-name=mnist-mps
#SBATCH --partition=teaching
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
```

Lines starting with `#SBATCH` look like comments to bash, but `sbatch` reads them as **resource requests**. Everything
you do not request, you do not get (a job without `--gres` gets no GPU; a job without `--mem` gets the cluster default of 4 GB per requested CPU).

| Option | Meaning |
|---|---|
| `--job-name` | label in `squeue`; also available as `%x` in file names |
| `--partition` | which queue (priority and time limit) the job goes to |
| `--gres=shard:1` | **one share of a GPU that is shared via MPS.** Other options: `gpu:mig_24gb:1`, `gpu:mig_48gb:1`, `gpu:full:1` |
| `--cpus-per-task` | CPU cores; `train.py` reads this to size its data-loader workers |
| `--mem` | RAM. Exceeding it kills the job; the VRAM share is a separate limit |
| `--time` | wall-clock limit (`HH:MM:SS`). Ask for a bit more than you need – the job is killed at the limit |
| `--output` | where stdout/stderr go; `%j` is the job id. **The directory must exist** |

```bash
cd "$SLURM_SUBMIT_DIR"
source ~/venvs/genai/bin/activate
nvidia-smi
srun python train.py --epochs 5
```

- A job starts in a **fresh shell** on the compute node, so it does not inherit what you did on the login node:
  we change to the submit directory and activate the venv explicitly.
- `nvidia-smi` is a cheap sanity check at the top of every log: which GPU did I get?
- `srun` launches the actual program as a job *step* (SLURM then tracks CPU/RAM usage for it in `sacct`).

## `setup_env.sh` – one-time preparation

`setup_env.sh` creates a virtual environment in `~/venvs/genai` and installs the packages of `../requirements.txt`.
It runs once on the **login node**; every job then just activates this environment.

## `train.py` – the program

### Block 1 – Arguments

```python
p.add_argument("--epochs", type=int, default=5) ...
```

All of this lives inside a function `main()` that is called under `if __name__ == "__main__":`. This is **required** as soon as
the `DataLoader` uses worker processes (as it does in the cluster job): recent Python versions start the workers by importing
`train.py` again, and without the guard each worker would start its own training run and crash with
*"An attempt has been made to start a new process before the current process has finished its bootstrapping phase"*.

Hyperparameters are command-line flags, so `job.sh` can change them without editing the code
(`srun python train.py --epochs 20`).

### Block 2 – Environment report

```python
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if ... else "cpu")
print(f"host=...  slurm_job_id=...  device={device}")
if device.type == "cuda":
    free, total = torch.cuda.mem_get_info()
```

The program picks the best available device, so the very same file runs on a laptop and on the cluster, and prints
where it runs. `SLURM_JOB_ID` is an environment variable set by SLURM inside a job (`-` on your laptop).

`torch.cuda.mem_get_info()` returns `(free, total)` in bytes: `total` is the whole card, `free` what is unused right now.
On an MPS-shared GPU the FAQ describes it as the way to see the VRAM share of your job (see the note in step 5 for what we
observed). `torch.cuda.get_device_properties(0).total_memory` and `nvidia-smi` always show the full card.

### Block 3 – Data

```python
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])
train_set = datasets.MNIST(args.data, train=True, download=True, transform=tf)
workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=workers)
```

- `ToTensor` converts the 28×28 image to a float tensor in [0, 1]; `Normalize(0.1307, 0.3081)` subtracts the mean and divides
  by the standard deviation of MNIST pixels – networks train faster on standardized inputs.
- `download=True` fetches MNIST into `../data` on the first run and does nothing afterwards (the data directory is shared by all jobs, as it lives in your home directory).
- `DataLoader` batches and shuffles; `num_workers` loads batches in parallel processes. On the cluster we use as
  many workers as CPUs requested with `--cpus-per-task`; on your laptop 0 (everything in the main process).

### Block 4 – Model

```python
class CNN(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 28 -> 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 14 -> 7
        return self.fc(x.flatten(1))
```

Two conv → ReLU → max-pool stages shrink the image from 28×28 to 7×7 while growing the channels from 1 to 32; a linear layer
maps the 32·7·7 = 1568 features to 10 class **logits** (no softmax – the loss applies it). About 20 k parameters (160 + 4 640 + 15 690).
`model.to(device)` moves the weights to the GPU.

### Block 5 – Training loop

```python
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    loss = F.cross_entropy(model(x), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

The standard PyTorch recipe: move the batch to the device, forward pass, loss, reset the old gradients, back-propagate,
update the weights. Then `model.eval()` and `torch.no_grad()` switch off training-only behaviour and gradient tracking
for the evaluation on the 10 000 test images.

`flush=True` in the `print` matters on the cluster: without it, output may sit in a buffer and your `tail -f` on the log
file stays empty for a long time.

### Block 6 – Saving

`torch.save(model.state_dict(), "outputs/mnist_cnn.pt")` stores the learned weights. Fetch them with `rsync` (step 6).

## Exemplary output

A run on a laptop (Apple GPU, `device=mps`; on HPC3 you will see `device=cuda` and the GPU line):

```
host=MacBook-Pro-von-Marc-2.local  slurm_job_id=-  device=mps
epoch 1/5  loss 0.0668  test acc 0.9776  (5.3s)
epoch 2/5  loss 0.1946  test acc 0.9821  (4.7s)
epoch 3/5  loss 0.0572  test acc 0.9858  (4.5s)
epoch 4/5  loss 0.0180  test acc 0.9850  (4.6s)
epoch 5/5  loss 0.0074  test acc 0.9873  (4.9s)
```

`loss` is the loss of the *last mini-batch* of the epoch (hence the bump in epoch 2); the **test accuracy** is the number
to watch: ~98 % after the first epoch, ~99 % after five.
