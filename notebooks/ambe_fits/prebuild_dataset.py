# %% [markdown]
# # Pre-build PyG dataset
# Loads the raw `.npy` file once and saves a list of `PYGData` objects to disk.
# Future training runs can `torch.load` instead of rebuilding every time.

# %%
import numpy as np
import torch
from torch_geometric.data import Data as PYGData

FLAVOUR = '6params'
FNAME = f'/home/puehlengt/appletree/notebooks/harvested_testsims_{FLAVOUR}.npy'
OUT   = f'/home/puehlengt/appletree/notebooks/pyg_dataset_raw_{FLAVOUR}.pt'

# %% [markdown]
# ## Load raw data

# %%
print("Loading...")
aa = np.load(FNAME, allow_pickle=True).item()
param_bag  = aa['param_bag']
events_bag = aa['events_bag']
N = len(events_bag)
print(f"{N} simulations found.")

# %% [markdown]
# ## Build PYGData objects
# Raw (cS1, cS2) — no log transform, no z-norm.
# `GraphData` in `train_deepset.py` applies those at runtime.

# %%
dataset = []
for i in range(N):
    _events = torch.tensor(events_bag[i].T, dtype=torch.float32)  # (n_events, 2)
    _params = torch.tensor(
        [v for v in param_bag[i].values()], dtype=torch.float32
    ).reshape(1, -1)                                               # (1, 3)
    dataset.append(PYGData(x=_events, y=_params))
    if i % 10000 == 0:
        print(f"{i}/{N}")

# %% [markdown]
# ## Save

# %%
torch.save(dataset, OUT)
print(f"Saved {N} samples to {OUT}.")

# %%
