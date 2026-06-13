"""
Hyperparameter search for AngleGNN using Optuna (TPE sampler).

Key features vs. the original version:
  - Explicit train/dev file lists instead of a random internal split, so the
    dev set is fixed and spans all structure types.
  - Edge-cutoff radius as an optional Optuna hyperparameter (--cutoff_values).
  - Graphs are pre-built at all candidate cutoffs at startup so each trial
    just indexes into a cached DataLoader.
  - Batch size is automatically capped to avoid CUDA OOM from edge_mlp
    intermediates (shape [n_edges, in_ch x out_ch]).
  - OOM safety catch: a trial that hits CUDA OOM returns float('inf') and the
    search continues rather than crashing.
  - --study {all,M2,M3} + --device_id let you run M2 and M3 in parallel on
    two GPUs:
      CUDA_VISIBLE_DEVICES=0 python optimize_hyperparameters.py --study M2 &
      CUDA_VISIBLE_DEVICES=1 python optimize_hyperparameters.py --study M3 &
  - Optional MLflow tracking (disabled with --no_mlflow).

Usage:
  python optimize_hyperparameters.py \\
      --train_list train_list.txt \\
      --dev_list   dev_list.txt \\
      --cutoff_values 3.0,4.0,5.0,6.0 \\
      --study all \\
      [--n_trials 100] [--epochs_per_trial 50] [--batch_size 256]
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime

import optuna
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader

from gnn_def import create_gnn_model_for_hyperparameter_search_variable_dims
from utils import load_gnn_samples_from_h5

INPUT_DIM = 14


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(tag, log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
    fpath = os.path.join(log_dir, f'optuna_{tag}_{ts}.log')
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(fpath), logging.StreamHandler()])
    return logging.getLogger(tag), fpath


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_file_list(path):
    with open(path) as f:
        return [line.split()[0] for line in f if line.strip()]


def build_cached_loaders(train_files, dev_files, cutoff_values, batch_size):
    """Pre-build DataLoaders at every candidate cutoff.

    Batch size is capped at 32: edge_mlp produces [n_edges, in_ch x out_ch]
    tensors kept alive for backprop. At r=6.0 A on 2x2 vac (1078 edges/graph),
    batch=32 gives ~34,500 edges; with hidden_dims_max=256 that is
    34,500 x 65,536 x 4 = 9 GB per layer — safe on a 40 GB A100.
    """
    safe_batch = min(batch_size, 32)
    train_loaders, dev_loaders = {}, {}
    for r in cutoff_values:
        print(f'  Building graphs at cutoff={r} A ...', flush=True)
        tr, dv = [], []
        for fp in train_files:
            tr.extend(load_gnn_samples_from_h5(fp, cutoff=r))
        for fp in dev_files:
            dv.extend(load_gnn_samples_from_h5(fp, cutoff=r))
        train_loaders[r] = DataLoader(tr, batch_size=safe_batch, shuffle=True)
        dev_loaders[r]   = DataLoader(dv, batch_size=32)
        print(f'    train: {len(tr)} states  |  dev: {len(dv)} states', flush=True)
    return train_loaders, dev_loaders


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(train_loaders, dev_loaders, epochs_per_trial,
                   cutoff_values, arch_params,
                   use_distances, use_angles, device, n_trials, logger):

    def objective(trial):
        cutoff = trial.suggest_categorical('cutoff', cutoff_values)
        lr     = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)

        model = create_gnn_model_for_hyperparameter_search_variable_dims(
            trial,
            input_dim=INPUT_DIM,
            params=arch_params,
            use_distances=use_distances,
            use_angles=use_angles,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn   = nn.MSELoss()
        t0        = time.time()

        try:
            for _ in range(epochs_per_trial):
                model.train()
                for batch in train_loaders[cutoff]:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    loss_fn(model(batch), batch.y.view(-1)).backward()
                    optimizer.step()

            model.eval()
            val_mae, val_n = 0.0, 0
            with torch.no_grad():
                for batch in dev_loaders[cutoff]:
                    batch    = batch.to(device)
                    val_mae += F.l1_loss(model(batch), batch.y.view(-1),
                                         reduction='sum').item()
                    val_n   += batch.num_graphs

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning(f'Trial {trial.number + 1}: CUDA OOM — returning inf')
            return float('inf')

        mae = val_mae / val_n
        dur = time.time() - t0

        try:
            best_so_far = trial.study.best_value * 1000
        except ValueError:
            best_so_far = float('nan')

        msg = (f'Trial {trial.number + 1}/{n_trials} | cutoff={cutoff} A | '
               f'MAE={mae * 1000:.2f} meV | best={best_so_far:.2f} meV | {dur:.0f}s')
        logger.info(msg)
        print(msg, flush=True)
        return mae

    return objective


# ---------------------------------------------------------------------------
# Run one study
# ---------------------------------------------------------------------------

def run_study(tag, use_distances, use_angles,
              train_loaders, dev_loaders, cutoff_values, arch_params,
              n_trials, epochs_per_trial, device, seed, out_dir,
              mlflow_run=None):

    logger, log_path = setup_logging(tag)
    print(f'\n{"=" * 60}')
    print(f'Study: {tag}  |  {n_trials} trials x {epochs_per_trial} epochs')
    print(f'Log: {log_path}')
    print(f'{"=" * 60}')

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction='minimize', sampler=sampler)
    study.set_user_attr('total_trials', n_trials)

    obj = make_objective(train_loaders, dev_loaders, epochs_per_trial,
                         cutoff_values, arch_params,
                         use_distances, use_angles, device, n_trials, logger)
    study.optimize(obj, n_trials=n_trials)

    best = study.best_trial
    print(f'\n{tag} best MAE: {best.value * 1000:.2f} meV  (trial {best.number + 1})')
    print(f'Best params: {best.params}')

    best_params = {**best.params, 'use_distances': use_distances, 'use_angles': use_angles}
    out_path = os.path.join(out_dir, f'best_params_{tag}.json')
    with open(out_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f'Saved -> {out_path}')

    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_metric(f'best_mae_{tag}', best.value)
            mlflow.log_param(f'best_trial_{tag}', best.number + 1)
            for k, v in best.params.items():
                mlflow.log_param(f'best_{tag}_{k}', v)
            mlflow.log_artifact(out_path)
        except Exception:
            pass

    # Optuna visualization (optional dependency: plotly)
    try:
        import optuna.visualization as vis
        vis.plot_optimization_history(study).write_html(
            os.path.join(out_dir, f'opt_history_{tag}.html'))
        vis.plot_param_importances(study).write_html(
            os.path.join(out_dir, f'param_importances_{tag}.html'))
    except Exception:
        pass

    return study


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list',       required=True,
                        help='File listing training HDF5 paths, one per line')
    parser.add_argument('--dev_list',         required=True,
                        help='File listing dev/validation HDF5 paths, one per line')
    parser.add_argument('--cutoff_values',    default='3.0',
                        help='Comma-separated edge-cutoff candidates in Å '
                             '(e.g. "3.0,4.0,5.0,6.0"). '
                             'Single value = fixed cutoff, no hyperparameter.')
    parser.add_argument('--n_trials',         type=int, default=100)
    parser.add_argument('--epochs_per_trial', type=int, default=50)
    parser.add_argument('--batch_size',       type=int, default=256,
                        help='Capped at 32 internally to avoid CUDA OOM')
    parser.add_argument('--n_layers_min',     type=int, default=2)
    parser.add_argument('--n_layers_max',     type=int, default=3)
    parser.add_argument('--hidden_dims_min',  type=int, default=32)
    parser.add_argument('--hidden_dims_max',  type=int, default=256,
                        help='Keep <=256 to avoid CUDA OOM from edge_mlp '
                             'intermediates (shape [n_edges, in_ch x out_ch])')
    parser.add_argument('--hidden_dims_step', type=int, default=16)
    parser.add_argument('--dropout_rate_min', type=float, default=0.0)
    parser.add_argument('--dropout_rate_max', type=float, default=0.4)
    parser.add_argument('--mlp_hidden_min',   type=int, default=64)
    parser.add_argument('--mlp_hidden_max',   type=int, default=256)
    parser.add_argument('--seed',             type=int, default=42)
    parser.add_argument('--out_dir',          default='.')
    parser.add_argument('--study', choices=['all', 'M2', 'M3'], default='all',
                        help='Which study to run. Set CUDA_VISIBLE_DEVICES and '
                             '--device_id to run M2/M3 in parallel on 2 GPUs.')
    parser.add_argument('--device_id',        type=int, default=0,
                        help='CUDA device index (within CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--no_mlflow',        action='store_true',
                        help='Disable MLflow tracking')
    parser.add_argument('--mlflow_tracking_uri', default='sqlite:///mlflow.db')
    parser.add_argument('--mlflow_experiment', default='GNN_GW_hypersearch')
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device_id}')
        print(f'Device: cuda:{args.device_id} ({torch.cuda.get_device_name(args.device_id)})')
    else:
        device = torch.device('cpu')
        print('Device: cpu')

    # Cutoff values
    cutoff_values = [float(x) for x in args.cutoff_values.split(',')]
    print(f'Cutoff candidates: {cutoff_values} A')

    # Arch search params
    arch_params = dict(
        n_layers_min=args.n_layers_min,
        n_layers_max=args.n_layers_max,
        hidden_dims_min=args.hidden_dims_min,
        hidden_dims_max=args.hidden_dims_max,
        hidden_dims_step=args.hidden_dims_step,
        dropout_rate_min=args.dropout_rate_min,
        dropout_rate_max=args.dropout_rate_max,
        mlp_hidden_min=args.mlp_hidden_min,
        mlp_hidden_max=args.mlp_hidden_max,
    )

    train_files = read_file_list(args.train_list)
    dev_files   = read_file_list(args.dev_list)
    print(f'Train: {len(train_files)} files  |  Dev: {len(dev_files)} files')

    print('\nPre-building graphs at all cutoffs ...')
    train_loaders, dev_loaders = build_cached_loaders(
        train_files, dev_files, cutoff_values, args.batch_size)

    os.makedirs(args.out_dir, exist_ok=True)

    # MLflow
    mlflow_run = None
    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
            mlflow.set_experiment(args.mlflow_experiment)
            mlflow_run = mlflow.start_run(
                run_name=f'hpsearch_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            mlflow.log_params({
                'study': args.study,
                'cutoff_values': args.cutoff_values,
                'n_trials': args.n_trials,
                'epochs_per_trial': args.epochs_per_trial,
                **{k: v for k, v in arch_params.items()},
            })
        except Exception as e:
            print(f'MLflow unavailable: {e}')
            mlflow_run = None

    common = dict(
        train_loaders=train_loaders, dev_loaders=dev_loaders,
        cutoff_values=cutoff_values, arch_params=arch_params,
        n_trials=args.n_trials, epochs_per_trial=args.epochs_per_trial,
        device=device, out_dir=args.out_dir, mlflow_run=mlflow_run,
    )

    if args.study in ('all', 'M2'):
        run_study('M2', use_distances=True, use_angles=False,
                  seed=args.seed, **common)

    if args.study in ('all', 'M3'):
        run_study('M3', use_distances=True, use_angles=True,
                  seed=args.seed + 1, **common)

    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass

    print('\n=== All done ===')
