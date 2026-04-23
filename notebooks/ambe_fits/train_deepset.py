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
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

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


class GraphData(data.Dataset):
    def __init__(self, data, events_mean, events_std):
        self.data = data
        self.events_mean = events_mean
        self.events_std = events_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = item.x.float()
        x = torch.stack([x[:, 0], torch.log(x[:, 1])], dim=1)
        x = (x - self.events_mean) / self.events_std
        return PYGData(x=x, y=item.y)


class DeepSet(nn.Module):
    def __init__(self, in_channels, hidden_layers, hidden_channels, out_channels,
                 n_events_mean, n_events_std):
        super().__init__()
        layers = [nn.Linear(in_channels, hidden_channels)]
        for _ in range(hidden_layers - 1):
            layers += [nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)]
        self.node_mlp = nn.Sequential(*layers)

        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )
        self.register_buffer('n_events_mean', torch.tensor(n_events_mean, dtype=torch.float32))
        self.register_buffer('n_events_std',  torch.tensor(n_events_std,  dtype=torch.float32))

    def forward(self, x):
        x = x.to(self.n_events_mean.device)
        node_features, batch = x.x, x.batch
        node_embed = self.node_mlp(node_features)
        mean_pool = global_mean_pool(node_embed, batch)
        max_pool  = global_max_pool(node_embed, batch)
        ones = torch.ones(node_features.shape[0], 1, device=node_features.device)
        n_events = global_add_pool(ones, batch)
        n_events_norm = (n_events - self.n_events_mean) / self.n_events_std
        return self.global_mlp(torch.cat([mean_pool, max_pool, n_events_norm], dim=1))


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
    def __init__(self, *args, checkpoint_every=10, **kwargs):
        super().__init__(*args, **kwargs)
        self._epoch = 0
        self._checkpoint_every = checkpoint_every

    def _train_epoch(self, model, train_loader, val_loader, stepper):
        loss_train, loss_val = super()._train_epoch(model, train_loader, val_loader, stepper)
        wandb.log({'train_log_prob': -loss_train, 'val_log_prob': -loss_val})
        self._epoch += 1
        if self._epoch % self._checkpoint_every == 0:
            torch.save(model.state_dict(),
                       self.out_dir / f'checkpoint_epoch{self._epoch}.pt')
        return loss_train, loss_val


def main():
    with open("deepset_config.yaml") as f:
        cfg = yaml.safe_load(f)

    wandb.init(config=cfg, project=cfg['wandb']['project'], entity=cfg['wandb']['entity'])

    hs = wandb.config.get('hidden_size',   cfg['embedding']['hidden_size'])
    hl = wandb.config.get('hidden_layers', cfg['embedding']['hidden_layers'])
    cfg['embedding']['hidden_size']   = hs
    cfg['embedding']['hidden_layers'] = hl

    out_dir = (f"{cfg['save_base']}/"
               f"{cfg['data']['num_samples']}totalsamples_deepset_nsf"
               f"_h{cfg['model']['hidden_features']}_t{cfg['model']['num_transforms']}"
               f"_emb_hs{hs}_hl{hl}"
               f"_lr{cfg['training']['learning_rate']}")
    wandb.config.update({'out_dir': out_dir}, allow_val_change=True)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda:0')

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

    # --- Z-norm stats from training set ---
    _train_events = []
    for idx in idx_train:
        x = dataset[idx].x.float()
        _train_events.append(torch.stack([x[:, 0], torch.log(x[:, 1])], dim=1))
    _train_events = torch.cat(_train_events, dim=0)
    events_mean = _train_events.mean(dim=0)
    events_std  = _train_events.std(dim=0).clamp(min=1e-8)

    _train_n = torch.tensor([float(dataset[idx].x.shape[0]) for idx in idx_train])
    n_events_mean = _train_n.mean().item()
    n_events_std  = _train_n.std().clamp(min=1e-8).item()

    with open(f'{out_dir}/norm_stats.pkl', 'wb') as f:
        pickle.dump({
            'events_mean': events_mean, 'events_std': events_std,
            'n_events_mean': n_events_mean, 'n_events_std': n_events_std,
        }, f)

    # --- DataLoaders ---
    graph_dataset = GraphData(dataset, events_mean, events_std)
    collater = Collater(dataset=graph_dataset, follow_batch='y')

    def collate_fn(batch):
        batch = collater(batch).to(device)
        return PyGBatchWrapper(batch), batch.y

    bs = cfg['training']['batch_size']
    train_loader = data.DataLoader(
        graph_dataset, batch_size=bs, collate_fn=collate_fn,
        sampler=data.SubsetRandomSampler(idx_train), drop_last=True,
    )
    val_loader = data.DataLoader(
        graph_dataset, batch_size=bs, collate_fn=collate_fn,
        sampler=data.SubsetRandomSampler(idx_val),
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
    embedding = DeepSet(
        in_channels=cfg['embedding']['in_channels'],
        hidden_layers=hl,
        hidden_channels=hs,
        out_channels=cfg['embedding']['output_size'],
        n_events_mean=n_events_mean,
        n_events_std=n_events_std,
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
            'lr_scheduler': cfg['training'].get('lr_scheduler', 'ReduceLROnPlateau'),
            'lr_decay_factor': cfg['training'].get('lr_decay_factor', 1),
            'lr_patience': cfg['training'].get('lr_patience', 10),
        },
        proposal=None,
        out_dir=out_dir,
    )
    posterior_ensemble, _ = runner(loader=loader)

    # --- Evaluate on test2 ---
    theta_test2 = np.array([d.y.numpy().flatten() for d in dataset_test2])
    test2_graph = GraphData(dataset_test2, events_mean, events_std)

    param_names = list(apt_param_config.keys())
    metric = PosteriorCoverage(
        num_samples=1000, sample_method='direct',
        out_dir=out_dir, labels=param_names,
        plot_list=['coverage', 'histogram', 'predictions', 'tarp'],
        save_samples=True,
    )
    metric(posterior=posterior_ensemble, x=test2_graph, theta=theta_test2)

    wandb.finish()


if __name__ == '__main__':
    main()
