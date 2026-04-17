import argparse
import os
import pickle
import yaml
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils import data

import ili
from ili.dataloaders import TorchLoader
from ili.inference.runner_lampe import LampeRunner
from ili.validation.metrics import PosteriorCoverage

apt_param_config = {
    "g1": {
        "prior_type": "norm",
        "prior_args": {"mean": 0.1367, "std": 0.001},
        "allowed_range": [0, 1.0],
        "init_mean": 0.1367, "init_std": 0.001,
        "unit": "PE/photon", "doc": "g1"
    },
    "g2": {
        "prior_type": "norm",
        "prior_args": {"mean": 16.85, "std": 0.46},
        "allowed_range": [0, 100.0],
        "init_mean": 16.85, "init_std": 0.46,
        "unit": "PE/electron", "doc": "g2"
    },
    "ambe_nr_rate": {
        "prior_type": "free",
        "prior_args": {},
        "allowed_range": [0, 10000000000.0],
        "init_mean": 5500, "init_std": 100,
        "unit": "1", "doc": "total number of events in the AmBe NR calibration"
    }
}


def make_summary(events):
    cs1 = events[0]
    log_cs2 = np.log10(events[1])
    n_events = len(cs1)
    return np.array([
        np.mean(cs1),
        np.std(cs1),
        np.mean(log_cs2),
        np.std(log_cs2),
        float(n_events),
    ], dtype=np.float32)


class SummaryDataset(data.Dataset):
    def __init__(self, x, theta):
        self.x = x
        self.theta = theta

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.theta[idx]


class WandbLampeRunner(LampeRunner):
    def _train_epoch(self, model, train_loader, val_loader, stepper):
        loss_train, loss_val = super()._train_epoch(model, train_loader, val_loader, stepper)
        wandb.log({'train_log_prob': -loss_train, 'val_log_prob': -loss_val})
        return loss_train, loss_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_features', type=int, default=None)
    parser.add_argument('--num_transforms', type=int, default=None)
    args = parser.parse_args()

    with open("summary_config.yaml") as f:
        cfg = yaml.safe_load(f)

    wandb.init(config=cfg)

    hf = (wandb.config['hidden_features'] if 'hidden_features' in wandb.config
          else (args.hidden_features or cfg['model']['hidden_features']))
    nt = (wandb.config['num_transforms'] if 'num_transforms' in wandb.config
          else (args.num_transforms or cfg['model']['num_transforms']))
    cfg['model']['hidden_features'] = hf
    cfg['model']['num_transforms'] = nt

    cfg['out_dir'] = (
        f"{cfg['save_base']}/{cfg['data']['num_samples']}totalsamples_"
        f"summary_nsf_h{hf}_t{nt}"
    )
    wandb.config.update({'out_dir': cfg['out_dir']}, allow_val_change=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load data ---
    fname = f"{cfg['data']['base_dir']}/{cfg['data']['fname']}"
    aa = np.load(fname, allow_pickle=True).item()
    param_bag = aa['param_bag']
    events_bag = aa['events_bag']
    num_samples = cfg['data']['num_samples']

    # --- Split ---
    np.random.seed(42)
    n_test = 10
    validation_fraction = 0.1
    permuted_idx = np.random.permutation(num_samples)
    idx_test = permuted_idx[:n_test]
    idx_remaining = permuted_idx[n_test:]
    n_train = int((1 - validation_fraction) * len(idx_remaining))
    idx_train = idx_remaining[:n_train]
    idx_val = idx_remaining[n_train:]

    # --- Build summary stats ---
    x_data = []
    theta_data = []
    for idx in range(num_samples):
        x_data.append(make_summary(events_bag[idx]))
        theta_data.append(np.array([v for v in param_bag[idx].values()], dtype=np.float32))

    x_data = np.stack(x_data)
    theta_data = torch.tensor(np.stack(theta_data))

    # --- Z-norm from training set ---
    x_mean = x_data[idx_train].mean(axis=0)
    x_std = x_data[idx_train].std(axis=0)
    x_std = np.where(x_std == 0, 1.0, x_std)
    x_data = torch.tensor((x_data - x_mean) / x_std, dtype=torch.float32)

    os.makedirs(cfg['out_dir'], exist_ok=True)
    with open(f"{cfg['out_dir']}/x_norm_stats.pkl", 'wb') as f:
        pickle.dump({'x_mean': x_mean, 'x_std': x_std}, f)

    # --- DataLoaders ---
    bs = cfg['training']['batch_size']
    dataset = SummaryDataset(x_data, theta_data)
    train_loader = data.DataLoader(
        dataset, batch_size=bs,
        sampler=data.SubsetRandomSampler(idx_train), drop_last=True
    )
    val_loader = data.DataLoader(
        dataset, batch_size=bs,
        sampler=data.SubsetRandomSampler(idx_val)
    )
    loader = TorchLoader(train_loader, val_loader)

    # --- Prior ---
    means, stds, low_bounds, high_bounds = [], [], [], []
    for param_info in apt_param_config.values():
        means.append(param_info['prior_args'].get('mean', param_info['init_mean']))
        stds.append(param_info['prior_args'].get('std', param_info['init_std']))
        low_bounds.append(param_info['allowed_range'][0])
        high_bounds.append(param_info['allowed_range'][1])

    prior = ili.utils.distributions_pt.IndependentTruncatedNormal(
        loc=means, scale=stds, low=low_bounds, high=high_bounds, device=device
    )

    # --- NSF ---
    nets = [
        ili.utils.load_nde_lampe(
            model=cfg['model']['type'],
            hidden_features=hf,
            num_transforms=nt,
            embedding_net=nn.Identity(),
            x_normalize=False,
            theta_normalize=True,
            device=device,
        )
    ]

    # --- Train ---
    runner = WandbLampeRunner(
        prior=prior,
        nets=nets,
        device=device,
        train_args={
            'training_batch_size': bs,
            'learning_rate': cfg['training']['learning_rate'],
            'weight_decay': cfg['training']['weight_decay'],
            'stop_after_epochs': cfg['training']['stop_after_epochs'],
            'clip_max_norm': cfg['training']['clip_max_norm'],
            'max_epochs': cfg['training']['max_epochs'],
        },
        proposal=None,
        out_dir=cfg['out_dir'],
    )

    posterior_ensemble, summaries = runner(loader=loader)

    # --- Evaluate on test2 (fresh sims, never seen during training) ---
    N_TEST2 = 2000
    idx_test2 = np.arange(num_samples, num_samples + N_TEST2)

    x_test2 = []
    theta_test2 = []
    for idx in idx_test2:
        x_test2.append(make_summary(events_bag[idx]))
        theta_test2.append(np.array([v for v in param_bag[idx].values()], dtype=np.float32))

    x_test2 = torch.tensor((np.stack(x_test2) - x_mean) / x_std, dtype=torch.float32)
    theta_test2 = np.stack(theta_test2)

    class SummaryDatasetX(data.Dataset):
        def __init__(self, x):
            self.x = x
        def __len__(self):
            return len(self.x)
        def __getitem__(self, idx):
            return self.x[idx]

    param_names = list(apt_param_config.keys())
    metric = PosteriorCoverage(
        num_samples=1000, sample_method='direct',
        out_dir=cfg['out_dir'], labels=param_names,
        plot_list=['coverage', 'histogram', 'predictions', 'tarp'],
        save_samples=True,
    )
    metric(
        posterior=posterior_ensemble,
        x=SummaryDatasetX(x_test2), theta=theta_test2,
    )

    wandb.finish()


if __name__ == '__main__':
    main()
