import argparse
import os
import pickle
import yaml
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils import data
import matplotlib.pyplot as plt

import ili
from ili.dataloaders import TorchLoader
from ili.inference.runner_lampe import LampeRunner
from ili.validation.metrics import PosteriorCoverage

apt_param_config = {
    "g1": {
        "prior_args": {"mean": 0.1367, "std": 0.001},
        "init_mean": 0.1367, "init_std": 0.001,
        "allowed_range": [0, 1.0],
    },
    "g2": {
        "prior_args": {"mean": 16.85, "std": 0.46},
        "init_mean": 16.85, "init_std": 0.46,
        "allowed_range": [0, 100.0],
    },
    "ambe_nr_rate": {
        "prior_args": {},
        "init_mean": 5500, "init_std": 100,
        "allowed_range": [0, 10000000000.0],
    },
}


def make_histogram(events, cs1_edges, log_cs2_edges):
    """Compute density-normalised 2D histogram from raw events array [N, 2]."""
    cs1     = events[:, 0]
    log_cs2 = np.log10(events[:, 1])
    n_events = float(len(cs1))
    h, _, _ = np.histogram2d(cs1, log_cs2, bins=[cs1_edges, log_cs2_edges], density=True)
    return h.astype(np.float32), n_events


class CNNDataset(data.Dataset):
    """Returns packed [hist_flat (N_BINS^2), n_events_norm (1)] and optionally theta."""
    def __init__(self, x_packed, theta=None):
        self.x_packed = x_packed  # [N, N_BINS^2 + 1]
        self.theta = theta         # [N, 3] or None

    def __len__(self):
        return len(self.x_packed)

    def __getitem__(self, idx):
        if self.theta is None:
            return self.x_packed[idx]
        return self.x_packed[idx], self.theta[idx]


class CNNEmbedding(nn.Module):
    """CNN over 2D histogram + appended z-normed n_events.

    Input x is packed flat tensor: [hist_flat (n_bins^2), n_events_norm (1)].
    CNN processes the 2D histogram; n_events is appended to the output.
    NSF input dim: output_size + 1.
    """
    def __init__(self, n_bins, channels, output_size):
        super().__init__()
        self.n_bins = n_bins
        self.n_hist = n_bins * n_bins

        conv_layers = []
        in_ch = 1
        for out_ch in channels:
            conv_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)

        spatial = n_bins
        for _ in channels:
            spatial = spatial // 2
        flattened = in_ch * spatial * spatial

        self.fc = nn.Linear(flattened, output_size)

    def forward(self, x):
        # reshape handles both batched [B, n_hist+1] (training) and single [n_hist+1] (evaluation)
        x = x.to(next(self.parameters()).device).reshape(-1, self.n_hist + 1)
        hist = x[:, :self.n_hist].reshape(-1, 1, self.n_bins, self.n_bins)
        n_ev  = x[:, self.n_hist:]  # [B, 1]
        cnn_out = self.fc(self.conv(hist).flatten(1))
        return torch.cat([cnn_out, n_ev], dim=1)


class WandbLampeRunner(LampeRunner):
    def __init__(self, *args, checkpoint_every=10, **kwargs):
        super().__init__(*args, **kwargs)
        self._epoch = 0
        self._checkpoint_every = checkpoint_every

    def _train_epoch(self, model, train_loader, val_loader, stepper):
        loss_train, loss_val = super()._train_epoch(model, train_loader, val_loader, stepper)
        wandb.log({
            'train_log_prob': -loss_train,
            'val_log_prob': -loss_val,
            'lr': stepper.optimizer.param_groups[0]['lr'],
        })
        self._epoch += 1
        if self._epoch % self._checkpoint_every == 0:
            torch.save(model.state_dict(),
                       self.out_dir / f'checkpoint_epoch{self._epoch}.pt')
        return loss_train, loss_val


def main():
    with open("cnn_config.yaml") as f:
        cfg = yaml.safe_load(f)

    wandb.init(config=cfg, project=cfg['wandb']['project'], entity=cfg['wandb']['entity'])

    out_size = cfg['cnn']['output_size']
    n_bins   = cfg['histogram']['n_bins']

    out_dir = (f"{cfg['save_base']}/"
               f"{cfg['data']['num_samples']}totalsamples_cnn"
               f"_{n_bins}x{n_bins}_emb{out_size}"
               f"_nsf_h{cfg['model']['hidden_features']}_t{cfg['model']['num_transforms']}"
               f"_lr{cfg['training']['learning_rate']}"
               f"_{cfg['training']['lr_scheduler']}")
    wandb.config.update({'out_dir': out_dir}, allow_val_change=True)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(f"cuda:{cfg['hardware']['cuda_device']}")

    num_samples = cfg['data']['num_samples']
    N_TEST2 = 2000
    total = num_samples + N_TEST2

    # --- Load dataset ---
    fname = f"{cfg['data']['base_dir']}/{cfg['data']['fname']}"
    print(f"Loading {fname}...")
    aa = np.load(fname, allow_pickle=True).item()
    param_bag  = aa['param_bag']
    events_bag = aa['events_bag']

    params = np.array(
        [[v for v in param_bag[i].values()] for i in range(num_samples)],
        dtype=np.float32,
    )
    params_test2 = np.array(
        [[v for v in param_bag[i].values()] for i in range(num_samples, total)],
        dtype=np.float32,
    )

    # --- Split ---
    np.random.seed(42)
    n_test = 10
    validation_fraction = 0.1
    permuted_idx = np.random.permutation(num_samples)
    idx_remaining = permuted_idx[n_test:]
    n_train = int((1 - validation_fraction) * len(idx_remaining))
    idx_train = idx_remaining[:n_train]
    idx_val   = idx_remaining[n_train:]

    # --- Build histograms ---
    cs1_edges     = np.linspace(*cfg['histogram']['cs1_range'],     n_bins + 1)
    log_cs2_edges = np.linspace(*cfg['histogram']['log_cs2_range'], n_bins + 1)

    print("Computing histograms...")
    hists, n_events_list = [], []
    for i in range(num_samples):
        h, n_ev = make_histogram(events_bag[i].T, cs1_edges, log_cs2_edges)
        hists.append(h.flatten())
        n_events_list.append(n_ev)

    hists    = np.stack(hists)                            # [N, n_bins^2]
    n_events = np.array(n_events_list, dtype=np.float32)  # [N]

    # --- Z-norm n_events only (density=True already normalises histogram) ---
    n_events_mean = n_events[idx_train].mean()
    n_events_std  = n_events[idx_train].std().clip(min=1e-8)
    n_events_norm = (n_events - n_events_mean) / n_events_std

    # --- Pack [hist_flat, n_events_norm] ---
    x_packed = torch.tensor(
        np.concatenate([hists, n_events_norm[:, None]], axis=1), dtype=torch.float32
    )
    theta = torch.tensor(params, dtype=torch.float32)

    # --- Test2 histograms ---
    print("Computing test2 histograms...")
    hists_test2, n_events_test2_list = [], []
    for i in range(num_samples, total):
        h, n_ev = make_histogram(events_bag[i].T, cs1_edges, log_cs2_edges)
        hists_test2.append(h.flatten())
        n_events_test2_list.append(n_ev)
    hists_test2    = np.stack(hists_test2)
    n_events_test2 = np.array(n_events_test2_list, dtype=np.float32)
    n_events_norm_test2 = (n_events_test2 - n_events_mean) / n_events_std
    x_packed_test2 = torch.tensor(
        np.concatenate([hists_test2, n_events_norm_test2[:, None]], axis=1),
        dtype=torch.float32,
    )

    theta_mean = torch.tensor(params[idx_train].mean(axis=0), dtype=torch.float32)
    theta_std  = torch.tensor(params[idx_train].std(axis=0),  dtype=torch.float32).clamp(min=1e-16)

    with open(f'{out_dir}/norm_stats.pkl', 'wb') as f:
        pickle.dump({
            'n_events_mean': n_events_mean, 'n_events_std': n_events_std,
            'cs1_edges': cs1_edges, 'log_cs2_edges': log_cs2_edges,
            'theta_mean': theta_mean, 'theta_std': theta_std,
        }, f)

    # --- DataLoaders ---
    bs          = cfg['training']['batch_size']
    num_workers = cfg['hardware']['num_workers']
    dataset_train = CNNDataset(x_packed, theta)
    train_loader = data.DataLoader(
        dataset_train, batch_size=bs,
        sampler=data.SubsetRandomSampler(idx_train), drop_last=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = data.DataLoader(
        dataset_train, batch_size=bs,
        sampler=data.SubsetRandomSampler(idx_val),
        num_workers=num_workers, pin_memory=True,
    )
    loader = TorchLoader(train_loader, val_loader)

    # --- Prior ---
    means, stds, lows, highs = [], [], [], []
    for p in apt_param_config.values():
        means.append(p['prior_args'].get('mean', p['init_mean']))
        stds.append(p['prior_args'].get('std',  p['init_std']))
        lows.append(p['allowed_range'][0])
        highs.append(p['allowed_range'][1])
    prior = ili.utils.distributions_pt.IndependentTruncatedNormal(
        loc=means, scale=stds, low=lows, high=highs, device=device,
    )

    # --- Model ---
    embedding = CNNEmbedding(
        n_bins=n_bins,
        channels=cfg['cnn']['channels'],
        output_size=out_size,
    ).to(device)

    nets = [
        ili.utils.load_nde_lampe(
            model=cfg['model']['type'],
            hidden_features=cfg['model']['hidden_features'],
            num_transforms=cfg['model']['num_transforms'],
            embedding_net=embedding,
            x_normalize=False,
            theta_normalize=True,
            device=device,
        )
    ]

    # --- Train ---
    runner = WandbLampeRunner(
        prior=prior, nets=nets, device=device,
        checkpoint_every=cfg['training'].get('checkpoint_every', 10),
        train_args={
            'training_batch_size': bs,
            'learning_rate': cfg['training']['learning_rate'],
            'weight_decay': cfg['training']['weight_decay'],
            'stop_after_epochs': cfg['training']['stop_after_epochs'],
            'clip_max_norm': cfg['training']['clip_max_norm'],
            'max_epochs': cfg['training']['max_epochs'],
            'lr_scheduler': cfg['training'].get('lr_scheduler', 'CosineAnnealingLR'),
            'lr_decay_factor': cfg['training'].get('lr_decay_factor', 1),
            'lr_patience': cfg['training'].get('lr_patience', 10),
        },
        proposal=None,
        out_dir=out_dir,
    )
    posterior_ensemble, _ = runner(loader=loader)

    # --- Evaluate ---
    dataset_test2_cnn = CNNDataset(x_packed_test2)
    param_names = list(apt_param_config.keys())
    metric = PosteriorCoverage(
        num_samples=1000, sample_method='direct',
        out_dir=out_dir, labels=param_names,
        plot_list=['coverage', 'histogram', 'predictions', 'tarp'],
        save_samples=True,
    )
    metric(posterior=posterior_ensemble, x=dataset_test2_cnn, theta=params_test2)

    wandb.finish()


def plot_sample_histograms(n=5):
    with open("cnn_config.yaml") as f:
        cfg = yaml.safe_load(f)

    n_bins        = cfg['histogram']['n_bins']
    cs1_edges     = np.linspace(*cfg['histogram']['cs1_range'],     n_bins + 1)
    log_cs2_edges = np.linspace(*cfg['histogram']['log_cs2_range'], n_bins + 1)

    fname = f"{cfg['data']['base_dir']}/{cfg['data']['fname']}"
    print(f"Loading {fname}...")
    aa = np.load(fname, allow_pickle=True).item()
    param_bag  = aa['param_bag']
    events_bag = aa['events_bag']

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for i, ax in enumerate(axes):
        h, n_ev = make_histogram(events_bag[i].T, cs1_edges, log_cs2_edges)
        im = ax.pcolormesh(cs1_edges, log_cs2_edges, h.T, shading='flat')
        plt.colorbar(im, ax=ax, label='density')
        params = list(param_bag[i].values())
        ax.set_title(f"g1={params[0]:.4f}\ng2={params[1]:.2f}\nrate={params[2]:.0f}\nn={n_ev:.0f}", fontsize=8)
        ax.set_xlabel('cS1')
        ax.set_ylabel('log10(cS2)')

    plt.tight_layout()
    out = f"sample_histograms_{n_bins}x{n_bins}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', help='Plot sample histograms and exit')
    parser.add_argument('--plot-n', type=int, default=5)
    args = parser.parse_args()

    if args.plot:
        plot_sample_histograms(n=args.plot_n)
    else:
        main()
