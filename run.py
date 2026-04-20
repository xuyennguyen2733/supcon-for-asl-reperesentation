"""Orchestrate all experiments for the ASL SupCon project.

Detects available GPUs and runs experiments in parallel across them.
With 1 GPU, experiments run sequentially in priority order.
With N GPUs, up to N experiments run concurrently.

Usage:
    python run.py                            # run all experiments
    python run.py --epochs 50                # override epoch count
    python run.py --dry_run                  # print commands without running
    python run.py --only 1 4 5               # run specific experiments by number
    python run.py --tmux training            # run inside tmux session "training"
                                             #   auto-stops RunPod when done
"""

import os
import sys
import subprocess
import shlex
import argparse
import signal
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.join('experiments', 'trained_models')

# Signals that indicate user-initiated kill (Ctrl+C, terminal kill, etc.)



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
                sys.executable, '-u', 'train.py', '--resume',
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
                sys.executable, '-u', 'train.py', '--resume',
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
                sys.executable, '-u', 'train.py', '--resume',
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
                sys.executable, '-u', 'train.py', '--resume',
                '--epochs', str(epochs),
                '--save_dir', os.path.join(BASE_DIR, '4_triplet_supcon_ce'),
            ],
        },
        {
            'name': '5_triplet_rope_supcon_ce',
            'description': 'Proposed + RoPE positional encoding',
            'pretrain_cmd': None,
            'train_cmd': [
                sys.executable, '-u', 'train.py', '--resume',
                '--use_rope',
                '--epochs', str(epochs),
                '--save_dir', os.path.join(BASE_DIR, '5_triplet_rope_supcon_ce'),
            ],
        },
        {
            'name': '6_triplet_pt_supcon_ce',
            'description': 'Proposed + masked pose pre-training',
            'pretrain_cmd': [
                sys.executable, '-u', '-m', 'models.pretrain',
                '--epochs', str(pretrain_epochs),
                '--save_dir', os.path.join(BASE_DIR, '6_triplet_pt_supcon_ce'),
            ],
            'train_cmd': [
                sys.executable, '-u', 'train.py', '--resume',
                '--pretrained_path', os.path.join(BASE_DIR, '6_triplet_pt_supcon_ce', 'pretrained_encoder.pt'),
                '--epochs', str(epochs),
                '--save_dir', os.path.join(BASE_DIR, '6_triplet_pt_supcon_ce'),
            ],
        },
        {
            'name': '7_triplet_rope_pt_supcon_ce',
            'description': 'Proposed + RoPE + pre-training (full pipeline)',
            'pretrain_cmd': [
                sys.executable, '-u', '-m', 'models.pretrain',
                '--use_rope',
                '--epochs', str(pretrain_epochs),
                '--save_dir', os.path.join(BASE_DIR, '7_triplet_rope_pt_supcon_ce'),
            ],
            'train_cmd': [
                sys.executable, '-u', 'train.py', '--resume',
                '--use_rope',
                '--pretrained_path', os.path.join(BASE_DIR, '7_triplet_rope_pt_supcon_ce', 'pretrained_encoder.pt'),
                '--epochs', str(epochs),
                '--save_dir', os.path.join(BASE_DIR, '7_triplet_rope_pt_supcon_ce'),
            ],
        },
        {
            'name': '8_triplet_supcon_then_ce',
            'description': 'Two-stage SupCon (Khosla et al.): stage 1 SupCon-only pretrain, stage 2 frozen encoder + linear CE',
            'pretrain_cmd': [
                sys.executable, '-u', 'train.py', '--resume',
                '--supcon_only',
                '--epochs', str(pretrain_epochs),
                '--save_dir', os.path.join(BASE_DIR, '8_triplet_supcon_then_ce'),
            ],
            'train_cmd': [
                sys.executable, '-u', 'train.py', '--resume',
                '--ce_only', '--freeze_encoder',
                '--pretrained_path', os.path.join(BASE_DIR, '8_triplet_supcon_then_ce', 'pretrained_encoder.pt'),
                '--epochs', str(epochs),
                '--save_dir', os.path.join(BASE_DIR, '8_triplet_supcon_then_ce'),
            ],
        },
    ]
    return experiments


def _run_with_progress(cmd, env, log_file, tag, phase):
    """Run subprocess, log everything, print brief epoch progress to console."""
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)), text=True, bufsize=1,
    )
    for line in proc.stdout:
        log_file.write(line)
        log_file.flush()
        s = line.strip()
        if not s:
            continue
        # Epoch lines: print every 10th, first, and last
        if s.startswith('Epoch'):
            try:
                parts = s.split('|')[0].split('/')
                epoch = int(parts[0].replace('Epoch', '').strip())
                total = int(parts[1].strip())
                if epoch == 1 or epoch % 10 == 0 or epoch == total:
                    val = ''
                    if 'val top1' in s:
                        val = s[s.index('val top1'):]
                    print(f"  [{tag}] {phase} epoch {epoch}/{total}  {val}")
            except (ValueError, IndexError):
                pass
        elif 'best model' in s.lower():
            print(f"  [{tag}] {s.split('->')[1].strip() if '->' in s else s}")
        elif 'error' in s.lower() or 'traceback' in s.lower():
            print(f"  [{tag}] {s}")
    proc.wait()
    return proc.returncode


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
            print(f"  [{name}] Pre-training...")
            log_file.write(f"=== PRE-TRAINING: {name} ===\n")
            log_file.write(f"Command: {' '.join(experiment['pretrain_cmd'])}\n\n")
            log_file.flush()

            rc = _run_with_progress(experiment['pretrain_cmd'], env, log_file, name, 'pretrain')
            if rc != 0:
                return name, False, f"Pre-training failed (exit code {rc}). See {log_path}"

            log_file.write(f"\n\n")

        # Training step
        print(f"  [{name}] Training...")
        log_file.write(f"=== TRAINING: {name} ===\n")
        log_file.write(f"Command: {' '.join(experiment['train_cmd'])}\n\n")
        log_file.flush()

        rc = _run_with_progress(experiment['train_cmd'], env, log_file, name, 'train')
        if rc != 0:
            return name, False, f"Training failed (exit code {rc}). See {log_path}"

    return name, True, f"Done. Log: {log_path}"


def run_experiments(experiments, gpu_ids, jobs_per_gpu=1):
    """Run all experiments, returns True if all succeeded."""
    all_success = True
    os.makedirs(BASE_DIR, exist_ok=True)

    max_workers = len(gpu_ids) * jobs_per_gpu

    if max_workers <= 1:
        # Sequential
        gpu = gpu_ids[0]
        for i, exp in enumerate(experiments, 1):
            gpu_label = f"GPU {gpu}" if gpu is not None else "CPU"
            print(f"[{i}/{len(experiments)}] Starting: {exp['name']} ({gpu_label})")
            name, success, msg = run_experiment(exp, gpu)
            status = "PASS" if success else "FAIL"
            if not success:
                all_success = False
            print(f"[{i}/{len(experiments)}] {status}: {name} — {msg}\n")
    else:
        # Parallel: distribute experiments across GPUs, jobs_per_gpu each
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, exp in enumerate(experiments):
                gpu = gpu_ids[i % len(gpu_ids)]
                future = executor.submit(run_experiment, exp, gpu)
                futures[future] = (i + 1, exp['name'], gpu)

            for future in as_completed(futures):
                idx, exp_name, gpu = futures[future]
                name, success, msg = future.result()
                status = "PASS" if success else "FAIL"
                if not success:
                    all_success = False
                print(f"[{idx}/{len(experiments)}] {status} (GPU {gpu}): {name} — {msg}")

    return all_success


def stop_runpod():
    """Stop the current RunPod instance via runpodctl."""
    try:
        subprocess.run(['runpodctl', 'stop', 'pod'], check=True, timeout=30)
        print("RunPod pod stopped.")
    except FileNotFoundError:
        print("runpodctl not found — cannot auto-stop pod.")
        print("Stop manually at https://www.runpod.io/console/pods")
    except subprocess.CalledProcessError as e:
        print(f"Failed to stop pod: {e}")
    except subprocess.TimeoutExpired:
        print("Timed out trying to stop pod.")


def prompt_stop_pod(timeout_seconds=120):
    """Ask the user whether to stop the RunPod pod.

    Returns True if the pod should be stopped:
      - User types y/yes/Enter -> stop
      - No response within timeout_seconds -> stop
      - User types n/no -> don't stop
    """
    import select

    print(f"\nAll experiments finished.")
    print(f"Stop RunPod pod to avoid charges? [Y/n] (auto-stops in {timeout_seconds}s) ", end='', flush=True)

    try:
        # Use select for timeout on Unix; fall back to alarm-based approach
        if hasattr(select, 'select'):
            ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
            if ready:
                answer = sys.stdin.readline().strip().lower()
            else:
                print("\nNo response — stopping pod.")
                return True
        else:
            # Windows fallback: use signal.alarm if available, otherwise no timeout
            answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        # Non-interactive terminal or Ctrl+C during prompt
        print("\nNo input available — stopping pod.")
        return True

    if answer in ('n', 'no'):
        print("Pod will keep running.")
        return False

    # y, yes, empty string, or anything else -> stop
    return True


def launch_in_tmux(session_name, argv):
    """Re-launch this script inside a tmux session.

    The inner invocation runs without --tmux so it executes normally.
    """
    # Build the inner command: same args but without --tmux <session>
    inner_args = []
    skip_next = False
    for arg in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == '--tmux':
            skip_next = True  # skip the session name that follows
            continue
        inner_args.append(arg)

    inner_cmd = f'{shlex.quote(sys.executable)} {shlex.quote(argv[0])} {" ".join(shlex.quote(a) for a in inner_args)}'
    project_dir = os.path.dirname(os.path.abspath(__file__))

    # Wrap in a shell script that:
    # 1. Runs the experiments
    # 2. On natural exit (finish/crash): prompts to stop pod
    # 3. On user kill (Ctrl+C / SIGINT): just exits, does NOT stop pod
    # Thin wrapper: runs the Python script, which handles all
    # post-run logic (rclone, pod stop) in main().
    wrapper_script = f"""#!/bin/bash
cd {shlex.quote(project_dir)}
{inner_cmd}
"""

    wrapper_path = os.path.join(project_dir, '.tmux_run_wrapper.sh')
    with open(wrapper_path, 'w', newline='\n') as f:
        f.write(wrapper_script)
    os.chmod(wrapper_path, 0o755)

    # Create or attach to tmux session
    # Check if session already exists
    check = subprocess.run(['tmux', 'has-session', '-t', session_name],
                           capture_output=True)
    if check.returncode == 0:
        print(f"tmux session '{session_name}' already exists.")
        print(f"Attach with: tmux attach -t {session_name}")
        print(f"Or kill it first: tmux kill-session -t {session_name}")
        sys.exit(1)

    # Start tmux session detached — script runs in background
    subprocess.run([
        'tmux', 'new-session', '-d', '-s', session_name, f'bash {shlex.quote(wrapper_path)}'
    ], check=True)

    print(f"Experiments running in tmux session: {session_name}")
    print(f"  Attach:  tmux attach -t {session_name}")
    print(f"  Detach:  Ctrl+B then D")
    print(f"  Logs:    tail -f experiments/trained_models/*/train.log")


def main():
    parser = argparse.ArgumentParser(description='Run all ASL SupCon experiments')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs per experiment')
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Pre-training epochs')
    parser.add_argument('--dry_run', action='store_true', help='Print commands without running')
    parser.add_argument('--only', type=int, nargs='+', default=None,
                        help='Run only specific experiments by number (e.g. --only 1 4 5)')
    parser.add_argument('--jobs_per_gpu', type=int, default=1,
                        help='Number of experiments to run concurrently per GPU (default: 1)')
    parser.add_argument('--tmux', type=str, default=None, metavar='SESSION',
                        help='Run inside a tmux session (survives disconnects, auto-stops RunPod when done)')
    args = parser.parse_args()

    # If --tmux is set, re-launch inside tmux and exit
    if args.tmux:
        launch_in_tmux(args.tmux, sys.argv)
        return  # unreachable after execvp, but for clarity

    experiments = get_experiments(args.epochs, args.pretrain_epochs)

    # Filter experiments if --only is specified
    if args.only:
        selected = set(args.only)
        experiments = [e for i, e in enumerate(experiments, 1) if i in selected]
        if not experiments:
            print(f"No experiments matched --only {args.only}. Valid range: 1-8.")
            return

    # Detect GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs detected. Running on CPU (this will be slow).")
        gpu_ids = [None]
    else:
        gpu_ids = list(range(num_gpus))
        print(f"Detected {num_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in gpu_ids]}")

    total_workers = len(gpu_ids) * args.jobs_per_gpu
    # Print experiment plan
    print(f"\n{'='*70}")
    print(f"EXPERIMENT PLAN: {len(experiments)} experiments, {len(gpu_ids)} GPU(s), "
          f"{args.jobs_per_gpu} job(s)/GPU = {total_workers} concurrent")
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

    # Track if we were killed by user
    user_killed = False
    def handle_signal(signum, frame):
        nonlocal user_killed
        user_killed = True
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Run experiments
    print(f"\n{'='*70}")
    print("TRAINING MODELS FOR EXPERIMENTS")
    print(f"{'='*70}\n")

    try:
        all_success = run_experiments(experiments, gpu_ids, args.jobs_per_gpu)
    except KeyboardInterrupt:
        user_killed = True
        print("\n\nUser interrupted — stopping experiments.")
        all_success = False
    except Exception as e:
        all_success = False
        print(f"\n\nTraining crashed: {e}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Results saved to: {BASE_DIR}/")
    print(f"Each experiment folder contains:")
    print(f"  best_model.pt  — best checkpoint (by val top-1)")
    print(f"  train.log      — full training output")
    if any(e['pretrain_cmd'] for e in experiments):
        print(f"  pretrained_encoder.pt — pre-trained weights (experiments 6, 7, 8)")

    # Post-run logic depends on how we got here
    if user_killed:
        # User interrupted — do nothing, just exit
        print("\nUser interrupted — pod will keep running.")
        sys.exit(130)

    if all_success:
        # All experiments passed — sync to Drive
        print(f"\nSyncing trained models to Google Drive...")
        rclone_result = subprocess.run(
            ['rclone', '--config', '/workspace/rclone.conf', 'copy',
             'experiments/trained_models/', 'gdrive:adv-com-vis-final/experiments/trained_models', '--progress'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if rclone_result.returncode == 0:
            print("Drive sync complete.")
        else:
            print(f"Drive sync failed (exit code {rclone_result.returncode}). "
                  f"Files are still in experiments/trained_models/ locally.")
    else:
        # Some experiments crashed — skip download, output may be corrupted
        print("\nSome experiments failed. Skipping Drive sync — inspect outputs manually.")

    # Prompt to stop pod
    if prompt_stop_pod():
        stop_runpod()


if __name__ == '__main__':
    main()
