import os
import pickle
import yaml
import numpy as np
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
    "alpha": {
        "prior_args": {"mean": 11.0, "std": 2.0},
        "init_mean": 11.0, "init_std": 2.0,
        "allowed_range": [1e-10, 10000000000.0],
    },
    "beta": {
        "prior_args": {"mean": 1.1, "std": 0.05},
        "init_mean": 1.1, "init_std": 0.05,
        "allowed_range": [-10000000000.0, 10000000000.0],
    },
    "epsilon": {
        "prior_args": {"mean": 12.6, "std": 3.4},
        "init_mean": 12.6, "init_std": 3.4,
        "allowed_range": [1e-10, 10000000000.0],
    },
}


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
        self.data = data
        self.events_mean = events_mean
        self.events_std = events_std
        self.summary_norm = summary_norm  # shape [N, 5], already z-normed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = item.x.float()
        x = torch.stack([x[:, 0], torch.log10(x[:, 1])], dim=1)
        x = (x - self.events_mean) / self.events_std
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
        x = x.to(self._device_ref.device)
        node_features, batch = x.x, x.batch
        summary_norm = x.summary  # [B, 5], already z-normed

        node_embed = self.node_mlp(node_features)
        mean_pool  = global_mean_pool(node_embed, batch)
        max_pool   = global_max_pool(node_embed, batch)
        deepset_out = self.global_mlp(torch.cat([mean_pool, max_pool], dim=1))

        return torch.cat([deepset_out, summary_norm], dim=1)


class PyGBatchWrapper:
    """Wraps a PyG Batch to expose .dtype, which ndes_pt.__call__ expects on x_batch."""
    def __init__(self, batch):
        self._batch = batch
        self.dtype = batch.x.dtype

    def cpu(self):
        return self._batch.cpu()

    def __getattr__(self, name):
        return getattr(self._batch, name)


class WandbLampeRunner(LampeRunner):
    def __init__(self, *args, checkpoint_every=10, resume_epoch=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._epoch = resume_epoch
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
    with open("hybrid5_deepset_6params_config.yaml") as f:
        cfg = yaml.safe_load(f)

    wandb.init(config=cfg, project=cfg['wandb']['project'], entity=cfg['wandb']['entity'])

    hs = wandb.config.get('hidden_size',   cfg['embedding']['hidden_size'])
    hl = wandb.config.get('hidden_layers', cfg['embedding']['hidden_layers'])
    cfg['embedding']['hidden_size']   = hs
    cfg['embedding']['hidden_layers'] = hl

    out_dir = (f"{cfg['save_base']}/"
               f"{cfg['data']['num_samples']}totalsamples_hybrid5_deepset_nsf"
               f"_h{cfg['model']['hidden_features']}_t{cfg['model']['num_transforms']}"
               f"_emb_hs{hs}_hl{hl}"
               f"_lr{cfg['training']['learning_rate']}"
               f"_{cfg['training']['lr_scheduler']}")
    wandb.config.update({'out_dir': out_dir}, allow_val_change=True)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(f"cuda:{cfg['hardware']['cuda_device']}")

    num_samples = cfg['data']['num_samples']
    N_TEST2 = 2000

    # --- Load dataset ---
    prebuilt_path = f"{cfg['data']['base_dir']}/{cfg['data'].get('prebuilt_fname', '')}"
    if cfg['data'].get('prebuilt_fname') and os.path.exists(prebuilt_path):
        print(f"Loading pre-built dataset from {prebuilt_path}...")
        full_dataset = torch.load(prebuilt_path)
        dataset = full_dataset[:num_samples]
        dataset_test2 = full_dataset[num_samples:num_samples + N_TEST2]
    else:
        fname = f"{cfg['data']['base_dir']}/{cfg['data']['fname']}"
        aa = np.load(fname, allow_pickle=True).item()
        param_bag  = aa['param_bag']
        events_bag = aa['events_bag']
        dataset = []
        for i in range(num_samples):
            _events = torch.tensor(events_bag[i].T)
            _params = torch.tensor([v for v in param_bag[i].values()]).reshape(1, -1)
            dataset.append(PYGData(x=_events, y=_params))
        dataset_test2 = []
        for idx in range(num_samples, num_samples + N_TEST2):
            _events = torch.tensor(events_bag[idx].T)
            _params = torch.tensor([v for v in param_bag[idx].values()]).reshape(1, -1)
            dataset_test2.append(PYGData(x=_events, y=_params))

    params = np.array([d.y.numpy().flatten() for d in dataset])

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
    summary_mean = all_summaries[idx_train].mean(axis=0)
    summary_std  = all_summaries[idx_train].std(axis=0).clip(min=1e-8)
    summary_norm = torch.tensor(
        (all_summaries - summary_mean) / summary_std, dtype=torch.float32
    )

    all_summaries_test2 = np.stack([make_summary(d) for d in dataset_test2])
    summary_norm_test2 = torch.tensor(
        (all_summaries_test2 - summary_mean) / summary_std, dtype=torch.float32
    )

    theta_mean = torch.tensor(params[idx_train].mean(axis=0), dtype=torch.float32)
    theta_std  = torch.tensor(params[idx_train].std(axis=0),  dtype=torch.float32).clamp(min=1e-16)

    with open(f'{out_dir}/norm_stats.pkl', 'wb') as f:
        pickle.dump({
            'events_mean': events_mean, 'events_std': events_std,
            'summary_mean': summary_mean, 'summary_std': summary_std,
            'theta_mean': theta_mean, 'theta_std': theta_std,
        }, f)

    # --- DataLoaders ---
    graph_dataset      = GraphData(dataset,       events_mean, events_std, summary_norm)
    graph_dataset_test2 = GraphData(dataset_test2, events_mean, events_std, summary_norm_test2)
    collater = Collater(dataset=graph_dataset, follow_batch='y')

    def collate_fn(batch):
        batch = collater(batch).to(device)
        return PyGBatchWrapper(batch), batch.y

    bs = cfg['training']['batch_size']
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

    # --- Resume from checkpoint ---
    resume_cfg = cfg.get('resume', {})
    checkpoint_path = resume_cfg.get('checkpoint')
    resume_epoch = int(resume_cfg.get('epoch', 0))
    if checkpoint_path:
        _orig_net = nets[0]
        def _make_resumed(factory, path, dev):
            def _resumed(train_loader, prior):
                model = factory(train_loader, prior)
                model.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
                return model
            return _resumed
        nets[0] = _make_resumed(_orig_net, checkpoint_path, device)

    # --- Train ---
    runner = WandbLampeRunner(
        prior=prior, nets=nets, device=device,
        checkpoint_every=cfg['training'].get('checkpoint_every', 10),
        resume_epoch=resume_epoch,
        train_args={
            'training_batch_size': bs,
            'learning_rate': cfg['training']['learning_rate'],
            'weight_decay': cfg['training']['weight_decay'],
            'stop_after_epochs': cfg['training']['stop_after_epochs'],
            'clip_max_norm': cfg['training']['clip_max_norm'],
            'max_epochs': cfg['training']['max_epochs'],
            'lr_scheduler': cfg['training'].get('lr_scheduler', 'ReduceLROnPlateau'),
            'lr_decay_factor': cfg['training'].get('lr_decay_factor', 0.5),
            'lr_patience': cfg['training'].get('lr_patience', 10),
        },
        proposal=None,
        out_dir=out_dir,
    )
    posterior_ensemble, _ = runner(loader=loader)

    # --- Evaluate on test2 ---
    theta_test2 = np.array([d.y.numpy().flatten() for d in dataset_test2])

    param_names = list(apt_param_config.keys())
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
