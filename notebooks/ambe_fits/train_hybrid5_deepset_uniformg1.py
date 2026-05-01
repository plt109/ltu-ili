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

from torch_geometric.data import Data as PYGData
from torch_geometric.loader.dataloader import Collater
from torch_geometric.nn import global_mean_pool, global_max_pool

import ili
from ili.dataloaders import TorchLoader
from ili.inference.runner_lampe import LampeRunner
from ili.validation.metrics import PosteriorCoverage
from ili.utils.distributions_pt import (
    IndependentTruncatedNormal,
    CustomJointIndependent,
    _UnivariateTruncatedNormal,
)


def build_prior_from_json(path, device):
    """Read a prior JSON and return a distribution suitable for ltu-ili.

    If all params are 'norm': returns a single vectorized IndependentTruncatedNormal
    (mirrors train_hybrid5_deepset.py; log_prob returns (B,) correctly).

    If mixed types: returns CustomJointIndependent with scalar components
    (_UnivariateTruncatedNormal / torch.distributions.Uniform, both event_shape=())
    so log_prob per component returns (B,) as CustomJointIndependent expects.
    """
    with open(path) as f:
        cfg = json.load(f)

    prior_types = [spec['prior_type'] for spec in cfg.values()]

    if all(pt == 'norm' for pt in prior_types):
        locs, scales, lows, highs = [], [], [], []
        for spec in cfg.values():
            args = spec['prior_args']
            low, high = spec['allowed_range']
            locs.append(args['mean'])
            scales.append(args['std'])
            lows.append(low)
            highs.append(high)
        return IndependentTruncatedNormal(
            loc=locs, scale=scales, low=lows, high=highs, device=device,
        )

    dists = []
    for par, spec in cfg.items():
        prior_type = spec['prior_type']
        args = spec['prior_args']
        low, high = spec['allowed_range']

        if prior_type == 'norm':
            loc   = torch.tensor([args['mean']], dtype=torch.float32, device=device)
            scale = torch.tensor([args['std']],  dtype=torch.float32, device=device)
            lo    = torch.tensor([low],           dtype=torch.float32, device=device)
            hi    = torch.tensor([high],          dtype=torch.float32, device=device)
            d = _UnivariateTruncatedNormal(loc=loc, scale=scale, low=lo, high=hi)
        elif prior_type == 'uniform':
            lo = torch.tensor(args['lower'], dtype=torch.float32, device=device)
            hi = torch.tensor(args['upper'], dtype=torch.float32, device=device)
            d = torch.distributions.Uniform(low=lo, high=hi)
        else:
            raise ValueError(f"Unsupported prior_type '{prior_type}' for param '{par}'")
        dists.append(d)

    return CustomJointIndependent(dists)


def load_dataset_from_hdf5(fname, num_samples):
    """Load up to num_samples sims from an HDF5 file, return list of PYGData."""
    with h5py.File(fname, 'r') as f:
        param_names = [p.decode() if isinstance(p, bytes) else p
                       for p in f.attrs['param_names']]
        n_events_all = f['events/n_events'][:]
        n_use    = min(num_samples, len(n_events_all))
        n_events = n_events_all[:n_use]
        total_ev = int(n_events.sum())
        cs1    = f['events/cs1'][:total_ev]
        cs2    = f['events/cs2'][:total_ev]
        params = np.stack([f['params'][par][:n_use] for par in param_names], axis=1)

    splits   = np.cumsum(n_events)[:-1]
    cs1_list = np.split(cs1, splits)
    cs2_list = np.split(cs2, splits)

    dataset = []
    for i in range(n_use):
        events  = np.vstack([cs1_list[i], cs2_list[i]]).T        # (n_i, 2)
        _events = torch.tensor(events,    dtype=torch.float32)
        _params = torch.tensor(params[i], dtype=torch.float32).reshape(1, -1)
        dataset.append(PYGData(x=_events, y=_params))

    return dataset, param_names


def make_summary(pyg_item):
    """Compute 5 summary stats from a PYGData item (raw, un-normed events)."""
    x = pyg_item.x.float()
    cs1     = x[:, 0].numpy()
    log_cs2 = torch.log10(x[:, 1]).numpy()
    return np.array([
        np.mean(cs1),
        np.std(cs1),
        np.mean(log_cs2),
        np.std(log_cs2),
        float(len(cs1)),
    ], dtype=np.float32)


class GraphData(data.Dataset):
    def __init__(self, data, events_mean, events_std, summary_norm):
        self.data         = data
        self.events_mean  = events_mean
        self.events_std   = events_std
        self.summary_norm = summary_norm  # shape [N, 5], already z-normed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x    = item.x.float()
        x    = torch.stack([x[:, 0], torch.log10(x[:, 1])], dim=1)
        x    = (x - self.events_mean) / self.events_std
        summary = self.summary_norm[idx]  # shape [5]
        return PYGData(x=x, y=item.y, summary=summary.unsqueeze(0))


class Hybrid5DeepSet(nn.Module):
    """DeepSet (out_channels-dim) + 5 pre-normalised summary stats -> NSF.

    NSF input dim: out_channels + 5.
    Summary stats [mean(cS1), std(cS1), mean(log10_cS2), std(log10_cS2), n_events]
    are precomputed and z-normed outside the model; forward just concatenates them.
    """
    def __init__(self, in_channels, hidden_layers, hidden_channels, out_channels):
        super().__init__()
        layers = [nn.Linear(in_channels, hidden_channels)]
        for _ in range(hidden_layers - 1):
            layers += [nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)]
        self.node_mlp = nn.Sequential(*layers)

        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )
        self.register_buffer('_device_ref', torch.zeros(1))

    def forward(self, x):
        x            = x.to(self._device_ref.device)
        node_features, batch = x.x, x.batch
        summary_norm = x.summary  # [B, 5], already z-normed

        node_embed  = self.node_mlp(node_features)
        mean_pool   = global_mean_pool(node_embed, batch)
        max_pool    = global_max_pool(node_embed, batch)
        deepset_out = self.global_mlp(torch.cat([mean_pool, max_pool], dim=1))

        return torch.cat([deepset_out, summary_norm], dim=1)


class PyGBatchWrapper:
    """Wraps a PyG Batch to expose .dtype, which ndes_pt.__call__ expects on x_batch."""
    def __init__(self, batch):
        self._batch = batch
        self.dtype  = batch.x.dtype

    def cpu(self):
        return self._batch.cpu()

    def __getattr__(self, name):
        return getattr(self._batch, name)


class WandbLampeRunner(LampeRunner):
    def __init__(self, *args, checkpoint_every=10, **kwargs):
        super().__init__(*args, **kwargs)
        self._epoch            = 0
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
    with open("hybrid5_deepset_uniformg1_config.yaml") as f:
        cfg = yaml.safe_load(f)

    wandb.init(config=cfg, project=cfg['wandb']['project'], entity=cfg['wandb']['entity'])

    hs = wandb.config.get('hidden_size',   cfg['embedding']['hidden_size'])
    hl = wandb.config.get('hidden_layers', cfg['embedding']['hidden_layers'])
    cfg['embedding']['hidden_size']   = hs
    cfg['embedding']['hidden_layers'] = hl

    out_dir = (f"{cfg['save_base']}/"
               f"{cfg['data']['num_samples']}totalsamples_hybrid5_deepset_uniformg1proposal_nsf"
               f"_h{cfg['model']['hidden_features']}_t{cfg['model']['num_transforms']}"
               f"_emb_hs{hs}_hl{hl}"
               f"_lr{cfg['training']['learning_rate']}"
               f"_{cfg['training']['lr_scheduler']}")
    wandb.config.update({'out_dir': out_dir}, allow_val_change=True)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(f"cuda:{cfg['hardware']['cuda_device']}")

    num_samples = cfg['data']['num_samples']
    N_TEST2     = cfg['data'].get('n_test', 2000)

    # --- Load datasets ---
    train_fname = f"{cfg['data']['train_dir']}/{cfg['data']['train_fname']}"
    test_fname  = f"{cfg['data']['test_dir']}/{cfg['data']['test_fname']}"
    print(f'Loading training data from {train_fname} ({num_samples} sims)...')
    dataset, param_names = load_dataset_from_hdf5(train_fname, num_samples)
    print(f'Loading test data from {test_fname} ({N_TEST2} sims)...')
    dataset_test2, _     = load_dataset_from_hdf5(test_fname, N_TEST2)
    dataset_test2, param_names_chk     = load_dataset_from_hdf5(test_fname, N_TEST2)
    assert param_names==param_names_chk, 'Fatal: param_names from train and test datasets do not match!'

    params = np.array([d.y.numpy().flatten() for d in dataset])

    # --- Split ---
    np.random.seed(42)
    n_test             = 10
    validation_fraction = 0.1
    permuted_idx = np.random.permutation(num_samples)
    idx_test      = permuted_idx[:n_test]
    idx_remaining = permuted_idx[n_test:]
    n_train       = int((1 - validation_fraction) * len(idx_remaining))
    idx_train     = idx_remaining[:n_train]
    idx_val       = idx_remaining[n_train:]

    # --- Z-norm stats for DeepSet node features ---
    _train_events = []
    for idx in idx_train:
        x = dataset[idx].x.float()
        _train_events.append(torch.stack([x[:, 0], torch.log10(x[:, 1])], dim=1))
    _train_events = torch.cat(_train_events, dim=0)
    events_mean = _train_events.mean(dim=0)
    events_std  = _train_events.std(dim=0).clamp(min=1e-8)

    # --- Precompute and z-norm 5 summary stats ---
    all_summaries = np.stack([make_summary(d) for d in dataset])  # [N, 5]
    summary_mean  = all_summaries[idx_train].mean(axis=0)
    summary_std   = all_summaries[idx_train].std(axis=0).clip(min=1e-8)
    summary_norm  = torch.tensor(
        (all_summaries - summary_mean) / summary_std, dtype=torch.float32)

    all_summaries_test2 = np.stack([make_summary(d) for d in dataset_test2])
    summary_norm_test2  = torch.tensor(
        (all_summaries_test2 - summary_mean) / summary_std, dtype=torch.float32)

    theta_mean = torch.tensor(params[idx_train].mean(axis=0), dtype=torch.float32)
    theta_std  = torch.tensor(params[idx_train].std(axis=0),  dtype=torch.float32).clamp(min=1e-16)

    with open(f'{out_dir}/norm_stats.pkl', 'wb') as f:
        pickle.dump({
            'events_mean':  events_mean,  'events_std':  events_std,
            'summary_mean': summary_mean, 'summary_std': summary_std,
            'theta_mean':   theta_mean,   'theta_std':   theta_std,
        }, f)

    # --- DataLoaders ---
    graph_dataset       = GraphData(dataset,       events_mean, events_std, summary_norm)
    graph_dataset_test2 = GraphData(dataset_test2, events_mean, events_std, summary_norm_test2)
    collater = Collater(dataset=graph_dataset, follow_batch='y')

    def collate_fn(batch):
        batch = collater(batch).to(device)
        return PyGBatchWrapper(batch), batch.y

    bs          = cfg['training']['batch_size']
    num_workers = cfg['hardware']['num_workers']
    train_loader = data.DataLoader(
        graph_dataset, batch_size=bs, collate_fn=collate_fn,
        sampler=data.SubsetRandomSampler(idx_train), drop_last=True,
        num_workers=num_workers,
    )
    val_loader = data.DataLoader(
        graph_dataset, batch_size=bs, collate_fn=collate_fn,
        sampler=data.SubsetRandomSampler(idx_val),
        num_workers=num_workers,
    )
    loader = TorchLoader(train_loader, val_loader)

    # --- Priors ---
    # proposal: distribution used to generate training sims (wide, uniform g1)
    # prior:    inference prior (tight Gaussian g1)
    proposal = build_prior_from_json(cfg['data']['train_prior_json'], device)
    prior    = build_prior_from_json(cfg['data']['test_prior_json'],  device)

    # --- Model ---
    embedding = Hybrid5DeepSet(
        in_channels=cfg['embedding']['in_channels'],
        hidden_layers=hl,
        hidden_channels=hs,
        out_channels=cfg['embedding']['output_size'],
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

    # --- Evaluate on test set ---
    theta_test2 = np.array([d.y.numpy().flatten() for d in dataset_test2])

    metric = PosteriorCoverage(
        num_samples=1000, sample_method='direct',
        out_dir=out_dir, labels=param_names,
        plot_list=['coverage', 'histogram', 'predictions', 'tarp'],
        save_samples=True,
    )
    metric(posterior=posterior_ensemble, x=graph_dataset_test2, theta=theta_test2)

    wandb.finish()


if __name__ == '__main__':
    main()
