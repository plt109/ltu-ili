"""
Architecture scan: train DeepSet+NSF over hidden_size x hidden_layers combinations.
Scans hidden_size in [64, 128, 256] x hidden_layers in [2, 3] = 6 runs total.
Runs in parallel across GPUs (round-robin). Fixed at 7k sims.
Architecture: mean+max+logncount embedding (poke_embedding.ipynb canonical).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils import data
from itertools import product

from torch_geometric.data import Data as PYGData
from torch_geometric.loader.dataloader import Collater
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as sps

import wandb
import ili
from ili.dataloaders import TorchLoader
from ili.inference.runner_lampe import LampeRunner
from ili.validation.metrics import PosteriorCoverage

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH          = "/home/puehlengt/appletree/notebooks/harvested_testsims_3params.npy"
N_SAMPLES          = 7000
HIDDEN_SIZE_LIST   = [64, 128, 256]
HIDDEN_LAYERS_LIST = [2, 3]

EMBED_CFG_BASE = dict(in_channels=2, output_size=8)
MODEL_CFG      = dict(model='nsf', hidden_features=32, num_transforms=3)
TRAIN_CFG      = dict(batch_size=32, learning_rate=1e-4, stop_after_epochs=20, max_epochs=1000)

WANDB_CFG      = dict(project='my-sbi-project', entity='plt109', group='ambe-3param')

N_TEST              = 10
VALIDATION_FRACTION = 0.1
SPLIT_SEED          = 42

PRIOR_MEANS  = [0.1367, 16.85, 5500.0]
PRIOR_STDS   = [0.001,  0.46,  100.0]
PRIOR_LOWS   = [0.0,    0.0,   0.0]
PRIOR_HIGHS  = [1.0,    100.0, 1e10]

PARAM_NAMES = ['g1', 'g2', 'ambe_nr_rate']

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class DeepSet(nn.Module):
    def __init__(self, in_channels, hidden_layers, hidden_channels, out_channels):
        super().__init__()
        layers = [nn.Linear(in_channels, hidden_channels)]
        for _ in range(hidden_layers - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.node_mlp = nn.Sequential(*layers)

        # Global MLP: input is mean_pool + max_pool + log(n_events) = 2 * hidden_channels + 1
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        node_features, batch = x.x, x.batch
        node_embed = self.node_mlp(node_features)
        mean_pool  = global_mean_pool(node_embed, batch)
        max_pool   = global_max_pool(node_embed, batch)
        ones       = torch.ones(node_features.shape[0], 1, device=node_features.device)
        n_events   = global_add_pool(ones, batch)
        return self.global_mlp(torch.cat([mean_pool, max_pool, torch.log(n_events)], dim=1))


class GraphData(data.Dataset):
    def __init__(self, data, events_mean, events_std):
        self.data        = data
        self.events_mean = events_mean
        self.events_std  = events_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x    = item.x.float()
        x    = torch.stack([x[:, 0], torch.log(x[:, 1])], dim=1)
        x    = (x - self.events_mean) / self.events_std
        return PYGData(x=x, y=item.y)


class WandbLampeRunner(LampeRunner):
    def _train_epoch(self, model, train_loader, val_loader, stepper):
        loss_train, loss_val = super()._train_epoch(model, train_loader, val_loader, stepper)
        wandb.log({'train_log_prob': -loss_train, 'val_log_prob': -loss_val})
        return loss_train, loss_val


# ---------------------------------------------------------------------------
# Per-run training function
# ---------------------------------------------------------------------------

def train_one(hidden_size, hidden_layers, gpu_id, graph_dataset, idx_train, idx_val, idx_test, all_params):
    device   = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    tag      = f"h{hidden_size}_l{hidden_layers}_{N_SAMPLES}sims"
    out_dir  = f"3_param_trained_models/arch_scan_logncount_scalar/{tag}"
    run_name = f"nsf_t3_{tag}"
    print(f"[{tag}] starting on {device}")
    collater      = Collater(dataset=graph_dataset, follow_batch='y')

    def collate_fn(batch):
        batch = collater(batch)
        return batch, batch.y

    train_loader = data.DataLoader(
        graph_dataset, batch_size=TRAIN_CFG['batch_size'], collate_fn=collate_fn,
        sampler=data.SubsetRandomSampler(idx_train), drop_last=True,
    )
    val_loader = data.DataLoader(
        graph_dataset, batch_size=TRAIN_CFG['batch_size'], collate_fn=collate_fn,
        sampler=data.SubsetRandomSampler(idx_val),
    )
    loader = TorchLoader(train_loader, val_loader)

    # --- prior ---
    prior = ili.utils.distributions_pt.IndependentTruncatedNormal(
        loc=PRIOR_MEANS, scale=PRIOR_STDS,
        low=PRIOR_LOWS, high=PRIOR_HIGHS,
        device=device,
    )

    # --- model ---
    embedding = DeepSet(
        in_channels=EMBED_CFG_BASE['in_channels'],
        hidden_layers=hidden_layers,
        hidden_channels=hidden_size,
        out_channels=EMBED_CFG_BASE['output_size'],
    )
    nets = [ili.utils.load_nde_lampe(
        embedding_net=embedding,
        x_normalize=False,
        theta_normalize=True,
        device=device,
        **MODEL_CFG,
    )]

    train_args = {
        'training_batch_size': TRAIN_CFG['batch_size'],
        'learning_rate':       TRAIN_CFG['learning_rate'],
        'stop_after_epochs':   TRAIN_CFG['stop_after_epochs'],
        'max_epochs':          TRAIN_CFG['max_epochs'],
    }

    # --- wandb ---
    wandb.init(
        project=WANDB_CFG['project'],
        entity=WANDB_CFG['entity'],
        group=WANDB_CFG['group'],
        name=run_name,
        tags=['arch-scan', 'logncount-scalar'],
        config=dict(
            hidden_size=hidden_size, hidden_layers=hidden_layers,
            n_samples=N_SAMPLES, gpu_id=gpu_id,
            **MODEL_CFG, **TRAIN_CFG,
        ),
        mode='offline',
    )

    # --- train ---
    runner = WandbLampeRunner(
        prior=prior, nets=nets, device=device,
        train_args=train_args, proposal=None, out_dir=out_dir,
    )
    posterior_ensemble, _ = runner(loader=loader)

    # --- plots ---
    x_obs      = graph_dataset[idx_test[0]]
    theta_true = x_obs.y[0].numpy()

    torch.manual_seed(1234)
    samples = posterior_ensemble.sample((1000,), x_obs).cpu().numpy()

    # 1) 1D marginal posteriors
    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(15, 4))
    for ii, name in enumerate(PARAM_NAMES):
        col = all_params[:, ii]
        xx  = np.linspace(col.min(), col.max(), 100)
        axes[ii].hist(col, bins=50, density=True, histtype='step', label='sim params')
        axes[ii].hist(samples[:, ii], bins=50, density=True, histtype='step', label='posterior samples')
        axes[ii].plot(xx, sps.norm.pdf(xx, PRIOR_MEANS[ii], PRIOR_STDS[ii]), label='prior')
        axes[ii].axvline(theta_true[ii], color='red', ls='--', label='true')
        axes[ii].set_xlabel(name)
        axes[ii].set_ylabel('density')
        if ii == len(PARAM_NAMES) - 1:
            axes[ii].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.suptitle(f'1D posteriors — hidden={hidden_size}, layers={hidden_layers}, {N_SAMPLES} sims')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/marginal_posteriors.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2) PosteriorCoverage
    coverage_metric = PosteriorCoverage(
        num_samples=1000, sample_method='direct',
        out_dir=out_dir, labels=PARAM_NAMES,
        plot_list=["coverage", "histogram", "predictions", "tarp"],
    )
    coverage_metric(posterior=posterior_ensemble, x=graph_dataset, theta=all_params)

    wandb.finish()
    print(f"[{tag}] done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    combos = list(product(HIDDEN_SIZE_LIST, HIDDEN_LAYERS_LIST))
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPU(s). Launching {len(combos)} processes "
          f"({len(HIDDEN_SIZE_LIST)} sizes x {len(HIDDEN_LAYERS_LIST)} depths).")

    # --- load data once in main process ---
    print("Loading data...")
    aa         = np.load(DATA_PATH, allow_pickle=True).item()
    param_bag  = aa['param_bag']
    events_bag = aa['events_bag']

    subset = []
    for i in range(N_SAMPLES):
        _events = torch.tensor(events_bag[i].T)
        _params = torch.tensor([v for v in param_bag[i].values()]).reshape(1, -1)
        subset.append(PYGData(x=_events, y=_params))

    all_params = np.array([
        [v for v in param_bag[i].values()] for i in range(N_SAMPLES)
    ])

    # --- split (same for all runs) ---
    rng       = np.random.default_rng(SPLIT_SEED)
    perm      = rng.permutation(N_SAMPLES)
    idx_test  = perm[:N_TEST]
    idx_rest  = perm[N_TEST:]
    n_train   = int((1 - VALIDATION_FRACTION) * len(idx_rest))
    idx_train = idx_rest[:n_train]
    idx_val   = idx_rest[n_train:]

    # --- normalization (same for all runs) ---
    _train_events = []
    for idx in idx_train:
        x = subset[idx].x.float()
        _train_events.append(torch.stack([x[:, 0], torch.log(x[:, 1])], dim=1))
    _train_events = torch.cat(_train_events, dim=0)
    events_mean   = _train_events.mean(dim=0)
    events_std    = _train_events.std(dim=0).clamp(min=1e-8)

    graph_dataset = GraphData(subset, events_mean, events_std)
    print("Data loaded. Spawning processes...")

    mp.set_start_method('spawn')

    processes = []
    for i, (hidden_size, hidden_layers) in enumerate(combos):
        gpu_id = i % max(n_gpus, 1)
        p = mp.Process(
            target=train_one,
            args=(hidden_size, hidden_layers, gpu_id, graph_dataset, idx_train, idx_val, idx_test, all_params),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\nAll runs complete.")

# 6 April 2026, 11:08PM
