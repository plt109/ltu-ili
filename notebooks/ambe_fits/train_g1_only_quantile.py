import argparse
import os
import pickle
import yaml
import json
import numpy as np
import h5py
import wandb
import torch
import torch.nn as nn
from torch.utils import data

import ili
from ili.dataloaders import TorchLoader
from ili.inference.runner_lampe import LampeRunner
from ili.validation.metrics import PosteriorCoverage
from ili.utils.distributions_pt import (
    CustomJointIndependent,
    _UnivariateTruncatedNormal,
)

QUANTILES = np.linspace(0.1, 1.0, 10)  # 10th to 100th percentile


def build_prior_from_json(path, device):
    with open(path) as f:
        cfg = json.load(f)

    dists = []
    for par, spec in cfg.items():
        prior_type = spec['prior_type']
        args = spec['prior_args']
        low, high = spec['allowed_range']
        if prior_type == 'norm':
            d = _UnivariateTruncatedNormal(
                loc=torch.tensor([args['mean']], dtype=torch.float32, device=device),
                scale=torch.tensor([args['std']], dtype=torch.float32, device=device),
                low=torch.tensor([low], dtype=torch.float32, device=device),
                high=torch.tensor([high], dtype=torch.float32, device=device),
            )
        elif prior_type == 'uniform':
            d = torch.distributions.Uniform(
                low=torch.tensor(args['lower'], dtype=torch.float32, device=device),
                high=torch.tensor(args['upper'], dtype=torch.float32, device=device),
            )
        else:
            raise ValueError(f"Unsupported prior_type '{prior_type}' for param '{par}'")
        dists.append(d)

    return CustomJointIndependent(dists)


def load_dataset_from_hdf5(fname, num_samples):
    """Load sims from HDF5, compute 10 cS1 quantiles per sim."""
    with h5py.File(fname, 'r') as f:
        param_names = [p.decode() if isinstance(p, bytes) else p
                       for p in f.attrs['param_names']]
        n_events_all = f['events/n_events'][:]
        n_use = min(num_samples, len(n_events_all))
        n_events = n_events_all[:n_use]
        total_ev = int(n_events.sum())
        cs1 = f['events/cs1'][:total_ev]
        params = np.stack([f['params'][par][:n_use] for par in param_names], axis=1)

    splits = np.cumsum(n_events)[:-1]
    cs1_list = np.split(cs1, splits)

    summaries = np.array(
        [np.quantile(c, QUANTILES) for c in cs1_list], dtype=np.float32
    )
    return summaries, params.astype(np.float32), param_names


class SummaryDataset(data.Dataset):
    def __init__(self, x, theta):
        self.x = x
        self.theta = theta

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.theta[idx]


class XDataset(data.Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


class WandbLampeRunner(LampeRunner):
    def __init__(self, *args, checkpoint_every=10, **kwargs):
        super().__init__(*args, **kwargs)
        self._epoch = 0
        self._checkpoint_every = checkpoint_every

    def _train_epoch(self, model, train_loader, val_loader, stepper):
        loss_train, loss_val = super()._train_epoch(model, train_loader, val_loader, stepper)
        wandb.log({
            'train_log_prob': -loss_train,
            'val_log_prob':   -loss_val,
            'lr': stepper.optimizer.param_groups[0]['lr'],
        })
        self._epoch += 1
        if self._epoch % self._checkpoint_every == 0:
            torch.save(model.state_dict(),
                       self.out_dir / f'checkpoint_epoch{self._epoch}.pt')
        return loss_train, loss_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='g1_only_quantile_config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    wandb.init(config=cfg, project=cfg['wandb']['project'], entity=cfg['wandb']['entity'])

    out_dir = (
        f"{cfg['save_base']}/"
        f"{cfg['data']['num_samples']}totalsamples_g1only_{cfg['run_tag']}_nsf"
        f"_h{cfg['model']['hidden_features']}_t{cfg['model']['num_transforms']}"
        f"_lr{cfg['training']['learning_rate']}"
        f"_{cfg['training']['lr_scheduler']}"
    )
    wandb.config.update({'out_dir': out_dir}, allow_val_change=True)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(f"cuda:{cfg['hardware']['cuda_device']}")

    num_samples = cfg['data']['num_samples']
    n_test      = cfg['data'].get('n_test', 2000)

    # --- Load datasets ---
    train_fname = f"{cfg['data']['train_dir']}/{cfg['data']['train_fname']}"
    test_fname  = f"{cfg['data']['test_dir']}/{cfg['data']['test_fname']}"
    print(f'Loading training data from {train_fname} ({num_samples} sims)...')
    x_all, theta_all, param_names = load_dataset_from_hdf5(train_fname, num_samples)
    print(f'Loading test data from {test_fname} ({n_test} sims)...')
    x_test, theta_test, param_names_chk = load_dataset_from_hdf5(test_fname, n_test)
    assert param_names == param_names_chk, 'param_names mismatch between train and test!'

    # --- Split ---
    np.random.seed(42)
    n_holdout = 10
    validation_fraction = 0.1
    permuted_idx = np.random.permutation(num_samples)
    idx_remaining = permuted_idx[n_holdout:]
    n_tr = int((1 - validation_fraction) * len(idx_remaining))
    idx_train = idx_remaining[:n_tr]
    idx_val   = idx_remaining[n_tr:]

    # --- Z-norm from training set ---
    x_mean = x_all[idx_train].mean(axis=0)
    x_std  = x_all[idx_train].std(axis=0).clip(min=1e-8)
    x_norm      = torch.tensor((x_all  - x_mean) / x_std, dtype=torch.float32)
    x_test_norm = torch.tensor((x_test - x_mean) / x_std, dtype=torch.float32)

    theta_all  = torch.tensor(theta_all,  dtype=torch.float32)
    theta_test = theta_test.astype(np.float32)

    theta_mean = theta_all[idx_train].mean(dim=0)
    theta_std  = theta_all[idx_train].std(dim=0).clamp(min=1e-16)
    with open(f'{out_dir}/norm_stats.pkl', 'wb') as f:
        pickle.dump({
            'x_mean': x_mean, 'x_std': x_std,
            'theta_mean': theta_mean, 'theta_std': theta_std,
        }, f)

    # --- DataLoaders ---
    bs      = cfg['training']['batch_size']
    dataset = SummaryDataset(x_norm, theta_all)
    train_loader = data.DataLoader(
        dataset, batch_size=bs,
        sampler=data.SubsetRandomSampler(idx_train), drop_last=True,
    )
    val_loader = data.DataLoader(
        dataset, batch_size=bs,
        sampler=data.SubsetRandomSampler(idx_val),
    )
    loader = TorchLoader(train_loader, val_loader)

    # --- Priors ---
    proposal = build_prior_from_json(cfg['data']['train_prior_json'], device)
    prior    = build_prior_from_json(cfg['data']['test_prior_json'],  device)

    # --- NSF (no embedding — 10-dim quantile summary straight in) ---
    nets = [
        ili.utils.load_nde_lampe(
            model=cfg['model']['type'],
            hidden_features=cfg['model']['hidden_features'],
            num_transforms=cfg['model']['num_transforms'],
            embedding_net=nn.Identity(),
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
            'learning_rate':       cfg['training']['learning_rate'],
            'weight_decay':        cfg['training']['weight_decay'],
            'stop_after_epochs':   cfg['training']['stop_after_epochs'],
            'clip_max_norm':       cfg['training']['clip_max_norm'],
            'max_epochs':          cfg['training']['max_epochs'],
            'lr_scheduler':        cfg['training'].get('lr_scheduler', 'CosineAnnealingLR'),
            'lr_decay_factor':     cfg['training'].get('lr_decay_factor', 1),
            'lr_patience':         cfg['training'].get('lr_patience', 10),
        },
        proposal=proposal,
        out_dir=out_dir,
    )
    posterior_ensemble, _ = runner(loader=loader)

    # --- Evaluate ---
    metric = PosteriorCoverage(
        num_samples=1000, sample_method='direct',
        out_dir=out_dir, labels=param_names,
        plot_list=['coverage', 'histogram', 'predictions', 'tarp'],
        save_samples=True,
    )
    metric(posterior=posterior_ensemble, x=XDataset(x_test_norm), theta=theta_test)

    wandb.finish()


if __name__ == '__main__':
    main()
