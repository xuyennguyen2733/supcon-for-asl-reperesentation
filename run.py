"""Orchestrate all experiments for the ASL SupCon project.

Detects available GPUs and runs experiments in parallel across them.
With 1 GPU, experiments run sequentially in priority order.
With N GPUs, up to N experiments run concurrently.

Usage:
    python run.py                   # run all experiments
    python run.py --epochs 50       # override epoch count
    python run.py --dry_run         # print commands without running
    python run.py --only 1 4 5      # run specific experiments by number
"""

import os
import sys
import subprocess
import argparse
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.join('experiments', 'trained_models')


def get_experiments(epochs, pretrain_epochs):
    """Define all experiments in priority order.

    Each experiment is a dict with:
        name:         human-readable name (also used as folder name)
        pretrain_cmd: optional pre-training command (must finish before train_cmd)
        train_cmd:    the main training command
        description:  what this experiment tests
    """
    experiments = [
        # === Phase 1: Core 2x2 (Pose-Triplet x SupCon) ===
        {
            'name': '1_flat_ce',
            'description': 'Baseline: flat tokens + CE only',
            'pretrain_cmd': None,
            'train_cmd': [
                sys.executable, 'train.py',
                '--no-use_triplet', '--ce_only',
                '--epochs', str(epochs),
                '--save_dir', os.path.join(BASE_DIR, '1_flat_ce'),
            ],
        },
        {
            'name': '2_flat_supcon_ce',
            'description': 'Flat tokens + SupCon + CE (isolates SupCon)',
            'pretrain_cmd': None,
            'train_cmd': [
                sys.executable, 'train.py',
                '--no-use_triplet',
                '--epochs', str(epochs),
                '--save_dir', os.path.join(BASE_DIR, '2_flat_supcon_ce'),
            ],
        },
        {
            'name': '3_triplet_ce',
            'description': 'Pose-Triplet + CE only (isolates triplet)',
            'pretrain_cmd': None,
            'train_cmd': [
                sys.executable, 'train.py',
                '--ce_only',
                '--epochs', str(epochs),
                '--save_dir', os.path.join(BASE_DIR, '3_triplet_ce'),
            ],
        },
        {
            'name': '4_triplet_supcon_ce',
            'description': 'Pose-Triplet + SupCon + CE (proposed method)',
            'pretrain_cmd': None,
            'train_cmd': [
                sys.executable, 'train.py',
                '--epochs', str(epochs),
                '--save_dir', os.path.join(BASE_DIR, '4_triplet_supcon_ce'),
            ],
        },
        # === Phase 2: Enhancements on the proposed method ===
        {
            'name': '5_triplet_rope_supcon_ce',
            'description': 'Proposed + RoPE positional encoding',
            'pretrain_cmd': None,
            'train_cmd': [
                sys.executable, 'train.py',
                '--use_rope',
                '--epochs', str(epochs),
                '--save_dir', os.path.join(BASE_DIR, '5_triplet_rope_supcon_ce'),
            ],
        },
        {
            'name': '6_triplet_pt_supcon_ce',
            'description': 'Proposed + masked pose pre-training',
            'pretrain_cmd': [
                sys.executable, '-m', 'models.pretrain',
                '--epochs', str(pretrain_epochs),
                '--save_dir', os.path.join(BASE_DIR, '6_triplet_pt_supcon_ce'),
            ],
            'train_cmd': [
                sys.executable, 'train.py',
                '--pretrained_path', os.path.join(BASE_DIR, '6_triplet_pt_supcon_ce', 'pretrained_encoder.pt'),
                '--epochs', str(epochs),
                '--save_dir', os.path.join(BASE_DIR, '6_triplet_pt_supcon_ce'),
            ],
        },
        {
            'name': '7_triplet_rope_pt_supcon_ce',
            'description': 'Proposed + RoPE + pre-training (full pipeline)',
            'pretrain_cmd': [
                sys.executable, '-m', 'models.pretrain',
                '--use_rope',
                '--epochs', str(pretrain_epochs),
                '--save_dir', os.path.join(BASE_DIR, '7_triplet_rope_pt_supcon_ce'),
            ],
            'train_cmd': [
                sys.executable, 'train.py',
                '--use_rope',
                '--pretrained_path', os.path.join(BASE_DIR, '7_triplet_rope_pt_supcon_ce', 'pretrained_encoder.pt'),
                '--epochs', str(epochs),
                '--save_dir', os.path.join(BASE_DIR, '7_triplet_rope_pt_supcon_ce'),
            ],
        },
    ]
    return experiments


def run_experiment(experiment, gpu_id=None):
    """Run a single experiment (pretrain if needed, then train).

    Returns (name, success, message).
    """
    name = experiment['name']
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    save_dir = os.path.join(BASE_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'train.log')

    with open(log_path, 'w') as log_file:
        # Pre-training step (if needed)
        if experiment['pretrain_cmd']:
            log_file.write(f"=== PRE-TRAINING: {name} ===\n")
            log_file.write(f"Command: {' '.join(experiment['pretrain_cmd'])}\n\n")
            log_file.flush()

            result = subprocess.run(
                experiment['pretrain_cmd'], env=env,
                stdout=log_file, stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            if result.returncode != 0:
                return name, False, f"Pre-training failed (exit code {result.returncode}). See {log_path}"

            log_file.write(f"\n\n")

        # Training step
        log_file.write(f"=== TRAINING: {name} ===\n")
        log_file.write(f"Command: {' '.join(experiment['train_cmd'])}\n\n")
        log_file.flush()

        result = subprocess.run(
            experiment['train_cmd'], env=env,
            stdout=log_file, stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode != 0:
            return name, False, f"Training failed (exit code {result.returncode}). See {log_path}"

    return name, True, f"Done. Log: {log_path}"


def main():
    parser = argparse.ArgumentParser(description='Run all ASL SupCon experiments')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs per experiment')
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Pre-training epochs')
    parser.add_argument('--dry_run', action='store_true', help='Print commands without running')
    parser.add_argument('--only', type=int, nargs='+', default=None,
                        help='Run only specific experiments by number (e.g. --only 1 4 5)')
    args = parser.parse_args()

    experiments = get_experiments(args.epochs, args.pretrain_epochs)

    # Filter experiments if --only is specified
    if args.only:
        selected = set(args.only)
        experiments = [e for i, e in enumerate(experiments, 1) if i in selected]
        if not experiments:
            print(f"No experiments matched --only {args.only}. Valid range: 1-7.")
            return

    # Detect GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs detected. Running on CPU (this will be slow).")
        gpu_ids = [None]
    else:
        gpu_ids = list(range(num_gpus))
        print(f"Detected {num_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in gpu_ids]}")

    # Print experiment plan
    print(f"\n{'='*70}")
    print(f"EXPERIMENT PLAN: {len(experiments)} experiments, {len(gpu_ids)} GPU(s)")
    print(f"{'='*70}")
    for i, exp in enumerate(experiments, 1):
        pretrain_tag = " [+pretrain]" if exp['pretrain_cmd'] else ""
        print(f"  {i}. {exp['name']:40s} {exp['description']}{pretrain_tag}")

    if args.dry_run:
        print(f"\n{'='*70}")
        print("DRY RUN — Commands that would be executed:")
        print(f"{'='*70}")
        for exp in experiments:
            print(f"\n--- {exp['name']} ---")
            if exp['pretrain_cmd']:
                print(f"  pretrain: {' '.join(exp['pretrain_cmd'])}")
            print(f"  train:    {' '.join(exp['train_cmd'])}")
        return

    # Run experiments
    print(f"\n{'='*70}")
    print("RUNNING EXPERIMENTS")
    print(f"{'='*70}\n")

    os.makedirs(BASE_DIR, exist_ok=True)

    if len(gpu_ids) <= 1:
        # Single GPU / CPU: run sequentially in priority order
        gpu = gpu_ids[0]
        for i, exp in enumerate(experiments, 1):
            gpu_label = f"GPU {gpu}" if gpu is not None else "CPU"
            print(f"[{i}/{len(experiments)}] Starting: {exp['name']} ({gpu_label})")
            name, success, msg = run_experiment(exp, gpu)
            status = "PASS" if success else "FAIL"
            print(f"[{i}/{len(experiments)}] {status}: {name} — {msg}\n")
    else:
        # Multi-GPU: run experiments in parallel, one per GPU.
        # Experiments with pretrain dependencies are self-contained
        # (pretrain runs inside run_experiment before train), so they
        # can safely be parallelized.
        results = {}
        with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = {}
            for i, exp in enumerate(experiments):
                gpu = gpu_ids[i % len(gpu_ids)]
                future = executor.submit(run_experiment, exp, gpu)
                futures[future] = (i + 1, exp['name'], gpu)

            for future in as_completed(futures):
                idx, exp_name, gpu = futures[future]
                name, success, msg = future.result()
                status = "PASS" if success else "FAIL"
                print(f"[{idx}/{len(experiments)}] {status} (GPU {gpu}): {name} — {msg}")
                results[exp_name] = success

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Results saved to: {BASE_DIR}/")
    print(f"Each experiment folder contains:")
    print(f"  best_model.pt  — best checkpoint (by val top-1)")
    print(f"  train.log      — full training output")
    if any(e['pretrain_cmd'] for e in experiments):
        print(f"  pretrained_encoder.pt — pre-trained weights (experiments 6, 7)")


if __name__ == '__main__':
    main()
