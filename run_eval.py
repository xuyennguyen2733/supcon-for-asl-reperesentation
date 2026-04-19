"""Evaluate all trained models and produce comparison reports.

For each model:
  - eval_results.json     — full metrics (top-1/5, distance ratio, per-class)
  - predictions.csv       — per-sample predictions (for custom tables/graphs)
  - tsne.png              — t-SNE embedding visualization

Combined outputs:
  - experiments/evaluation/eval_summary.csv  — one row per model, all metrics
  - experiments/evaluation/eval_report.md    — human-readable comparison report

Usage:
    python run_eval.py                              # evaluate all models in experiments/trained_models/
    python run_eval.py --models path/to/model1/best_model.pt path/to/model2/best_model.pt
    python run_eval.py --tmux eval_session           # run in tmux with pod auto-stop
    python run_eval.py --dry_run                     # preview what would be evaluated
"""

import os
import sys
import csv
import json
import signal
import subprocess
import shlex
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from train import ASLKeypointDataset, collate_eval
from models.encoder import SignLanguageEncoder
from eval import (
    collect_predictions_and_embeddings,
    compute_accuracy,
    compute_per_class_accuracy,
    compute_distance_ratio,
    generate_tsne,
)

BASE_DIR = os.path.join('experiments', 'trained_models')
EVAL_DIR = os.path.join('experiments', 'evaluation')

# Map experiment folder names to architecture flags
EXPERIMENT_CONFIG = {
    '1_flat_ce':                  {'use_triplet': False, 'use_rope': False},
    '2_flat_supcon_ce':           {'use_triplet': False, 'use_rope': False},
    '3_triplet_ce':               {'use_triplet': True,  'use_rope': False},
    '4_triplet_supcon_ce':        {'use_triplet': True,  'use_rope': False},
    '5_triplet_rope_supcon_ce':   {'use_triplet': True,  'use_rope': True},
    '6_triplet_pt_supcon_ce':     {'use_triplet': True,  'use_rope': False},
    '7_triplet_rope_pt_supcon_ce':{'use_triplet': True,  'use_rope': True},
}

EXPERIMENT_DESCRIPTIONS = {
    '1_flat_ce':                   'Flat + CE (baseline)',
    '2_flat_supcon_ce':            'Flat + SupCon + CE',
    '3_triplet_ce':                'Triplet + CE',
    '4_triplet_supcon_ce':         'Triplet + SupCon + CE (proposed)',
    '5_triplet_rope_supcon_ce':    'Triplet + RoPE + SupCon + CE',
    '6_triplet_pt_supcon_ce':      'Triplet + Pre-trained + SupCon + CE',
    '7_triplet_rope_pt_supcon_ce': 'Triplet + RoPE + Pre-trained + SupCon + CE',
}


def infer_config(model_dir):
    """Infer architecture flags from experiment folder name."""
    folder_name = os.path.basename(model_dir)
    if folder_name in EXPERIMENT_CONFIG:
        return EXPERIMENT_CONFIG[folder_name]
    # Fallback: guess from folder name
    use_rope = 'rope' in folder_name.lower()
    use_triplet = 'flat' not in folder_name.lower()
    return {'use_triplet': use_triplet, 'use_rope': use_rope}


def get_description(model_dir):
    """Get human-readable description for an experiment."""
    folder_name = os.path.basename(model_dir)
    return EXPERIMENT_DESCRIPTIONS.get(folder_name, folder_name)


def discover_models(base_dir):
    """Find all best_model.pt files in the base directory."""
    models = []
    if not os.path.exists(base_dir):
        return models
    for name in sorted(os.listdir(base_dir)):
        checkpoint = os.path.join(base_dir, name, 'best_model.pt')
        if os.path.isfile(checkpoint):
            models.append(checkpoint)
    return models


def save_predictions_csv(logits, labels, label_names, save_path):
    """Save per-sample predictions to CSV for charting."""
    preds = logits.argmax(dim=1)
    probs = torch.softmax(logits, dim=1)
    top5_preds = logits.topk(min(5, logits.size(1)), dim=1)

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample_idx', 'true_label', 'true_label_name',
            'predicted_label', 'predicted_label_name', 'correct',
            'confidence', 'top5_correct',
            'top5_pred_1', 'top5_pred_2', 'top5_pred_3', 'top5_pred_4', 'top5_pred_5',
        ])
        for i in range(len(labels)):
            true_idx = labels[i].item()
            pred_idx = preds[i].item()
            confidence = probs[i, pred_idx].item()
            top5_idx = top5_preds.indices[i].tolist()
            top5_correct = true_idx in top5_idx

            # Pad top5 to always have 5 entries
            top5_names = [label_names[j] if j < len(label_names) else '' for j in top5_idx]
            while len(top5_names) < 5:
                top5_names.append('')

            writer.writerow([
                i, true_idx, label_names[true_idx],
                pred_idx, label_names[pred_idx], int(true_idx == pred_idx),
                f'{confidence:.6f}', int(top5_correct),
                *top5_names,
            ])


def evaluate_model(checkpoint_path, test_dataset, test_loader, device):
    """Evaluate a single model. Saves outputs to EVAL_DIR/{model_name}/."""
    model_src_dir = os.path.dirname(checkpoint_path)
    config = infer_config(model_src_dir)
    description = get_description(model_src_dir)
    num_classes = len(test_dataset.labels)

    tag = os.path.basename(model_src_dir)
    out_dir = os.path.join(EVAL_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"  [{tag}] Evaluating {description}...")

    # Build model
    model = SignLanguageEncoder(
        num_classes=num_classes,
        use_triplet=config['use_triplet'],
        use_rope=config['use_rope'],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Collect predictions and embeddings
    data = collect_predictions_and_embeddings(model, test_loader, device)

    # Metrics
    acc = compute_accuracy(data['logits'], data['labels'])
    per_class = compute_per_class_accuracy(data['logits'], data['labels'], test_dataset.labels)
    dist = compute_distance_ratio(data['embeddings'], data['labels'])

    # Save per-model outputs to experiments/evaluation/{model_name}/
    predictions_path = os.path.join(out_dir, 'predictions.csv')
    save_predictions_csv(data['logits'], data['labels'], test_dataset.labels, predictions_path)

    results = {
        'experiment': tag,
        'description': description,
        'checkpoint': checkpoint_path,
        'config': config,
        'epoch': checkpoint['epoch'],
        'val_top1': checkpoint['val_top1'],
        'test_top1': acc['top1'],
        'test_top5': acc['top5'],
        'intra_class_distance': dist['intra_class_distance'],
        'inter_class_distance': dist['inter_class_distance'],
        'distance_ratio': dist['ratio'],
        'per_class': per_class,
    }
    results_path = os.path.join(out_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    tsne_path = os.path.join(out_dir, 'tsne.png')
    generate_tsne(data['embeddings'], data['labels'], test_dataset.labels, tsne_path)

    print(f"  [{tag}] Done.")
    return results


def write_summary_csv(all_results, save_path):
    """Write one-row-per-model CSV for easy graphing."""
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'experiment', 'description', 'use_triplet', 'use_rope',
            'epoch', 'val_top1', 'test_top1', 'test_top5',
            'intra_class_dist', 'inter_class_dist', 'distance_ratio',
        ])
        for r in all_results:
            writer.writerow([
                r['experiment'], r['description'],
                r['config']['use_triplet'], r['config']['use_rope'],
                r['epoch'], f"{r['val_top1']:.4f}",
                f"{r['test_top1']:.4f}", f"{r['test_top5']:.4f}",
                f"{r['intra_class_distance']:.4f}", f"{r['inter_class_distance']:.4f}",
                f"{r['distance_ratio']:.4f}",
            ])


def write_report(all_results, save_path):
    """Write a human-readable Markdown comparison report."""
    lines = []
    lines.append('# Evaluation Report')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('')

    # Summary table
    lines.append('## Summary')
    lines.append('')
    lines.append('| # | Experiment | Test Top-1 | Test Top-5 | Dist Ratio | Val Top-1 |')
    lines.append('|---|---|---|---|---|---|')
    for r in all_results:
        lines.append(
            f"| {r['experiment'].split('_')[0]} "
            f"| {r['description']} "
            f"| {r['test_top1']:.4f} "
            f"| {r['test_top5']:.4f} "
            f"| {r['distance_ratio']:.4f} "
            f"| {r['val_top1']:.4f} |"
        )
    lines.append('')

    # Best model
    best = max(all_results, key=lambda r: r['test_top1'])
    lines.append(f"**Best test top-1:** {best['description']} ({best['test_top1']:.4f})")
    lines.append('')

    best_ratio = min(all_results, key=lambda r: r['distance_ratio'])
    lines.append(f"**Best embedding quality (lowest distance ratio):** "
                 f"{best_ratio['description']} ({best_ratio['distance_ratio']:.4f})")
    lines.append('')

    # Key comparisons
    results_by_name = {r['experiment']: r for r in all_results}

    comparisons = [
        ('1_flat_ce', '2_flat_supcon_ce', 'SupCon contribution (flat tokens)'),
        ('1_flat_ce', '3_triplet_ce', 'Triplet contribution (CE only)'),
        ('1_flat_ce', '4_triplet_supcon_ce', 'Combined (proposed vs baseline)'),
        ('2_flat_supcon_ce', '4_triplet_supcon_ce', 'Triplet added value on top of SupCon'),
        ('3_triplet_ce', '4_triplet_supcon_ce', 'SupCon added value on top of Triplet'),
        ('4_triplet_supcon_ce', '5_triplet_rope_supcon_ce', 'RoPE improvement'),
        ('4_triplet_supcon_ce', '6_triplet_pt_supcon_ce', 'Pre-training improvement'),
        ('4_triplet_supcon_ce', '7_triplet_rope_pt_supcon_ce', 'RoPE + Pre-training improvement'),
    ]

    lines.append('## Key Comparisons')
    lines.append('')
    lines.append('| Comparison | A Top-1 | B Top-1 | Delta | A Ratio | B Ratio |')
    lines.append('|---|---|---|---|---|---|')

    for a_name, b_name, label in comparisons:
        a = results_by_name.get(a_name)
        b = results_by_name.get(b_name)
        if a and b:
            delta = b['test_top1'] - a['test_top1']
            sign = '+' if delta >= 0 else ''
            lines.append(
                f"| {label} "
                f"| {a['test_top1']:.4f} "
                f"| {b['test_top1']:.4f} "
                f"| {sign}{delta:.4f} "
                f"| {a['distance_ratio']:.4f} "
                f"| {b['distance_ratio']:.4f} |"
            )
    lines.append('')

    # Per-class detail for best model
    lines.append(f"## Per-Class Accuracy (Best Model: {best['description']})")
    lines.append('')
    lines.append('| Class | Accuracy | Test Samples |')
    lines.append('|---|---|---|')
    for name, stats in sorted(best['per_class'].items(), key=lambda x: x[1]['accuracy']):
        lines.append(f"| {name} | {stats['accuracy']:.4f} | {stats['count']} |")
    lines.append('')

    # File listing
    lines.append('## Output Files')
    lines.append('')
    lines.append('Each model folder in `experiments/evaluation/{model_name}/` contains:')
    lines.append('- `eval_results.json` — full metrics')
    lines.append('- `predictions.csv` — per-sample predictions (for custom charts)')
    lines.append('- `tsne.png` — t-SNE embedding visualization')
    lines.append('')
    lines.append('Combined outputs in `experiments/evaluation/`:')
    lines.append('- `eval_summary.csv` — one row per model (for graphing)')
    lines.append('- `eval_report.md` — this report')

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))


def launch_in_tmux(session_name, argv):
    """Re-launch this script inside a tmux session with pod auto-stop."""
    inner_args = []
    skip_next = False
    for arg in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == '--tmux':
            skip_next = True
            continue
        inner_args.append(arg)

    inner_cmd = f'{shlex.quote(sys.executable)} {shlex.quote(argv[0])} {" ".join(shlex.quote(a) for a in inner_args)}'
    project_dir = os.path.dirname(os.path.abspath(__file__))

    wrapper_script = f"""#!/bin/bash
cd {shlex.quote(project_dir)}

USER_KILLED=0
trap 'USER_KILLED=1; exit 130' INT TERM

{inner_cmd}
EXIT_CODE=$?

if [ $USER_KILLED -eq 1 ]; then
    echo ""
    echo "User interrupted — pod will keep running."
    exit $EXIT_CODE
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "All evaluations completed successfully."
else
    echo ""
    echo "Evaluations finished with errors (exit code $EXIT_CODE)."
fi

# Copy results to Google Drive
echo ""
echo "Syncing evaluation results to Google Drive..."
echo 'alias rclone="rclone --config /workspace/rclone.conf"' >> ~/.bashrc
source ~/.bashrc
rclone --config /workspace/rclone.conf copy experiments/evaluation/ gdrive:adv-com-vis-final/evaluation --progress
RCLONE_EXIT=$?
if [ $RCLONE_EXIT -eq 0 ]; then
    echo "Drive sync complete."
else
    echo "Drive sync failed (exit code $RCLONE_EXIT). Files are still in experiments/ locally."
fi

echo ""
echo -n "Stop RunPod pod to avoid charges? [Y/n] (auto-stops in 120s) "
read -t 120 ANSWER
READ_EXIT=$?

if [ $READ_EXIT -ne 0 ]; then
    echo ""
    echo "No response — stopping pod."
    runpodctl stop pod 2>/dev/null || echo "runpodctl not found. Stop manually."
    exit $EXIT_CODE
fi

ANSWER=$(echo "$ANSWER" | tr '[:upper:]' '[:lower:]')
if [ "$ANSWER" = "n" ] || [ "$ANSWER" = "no" ]; then
    echo "Pod will keep running."
else
    echo "Stopping pod..."
    runpodctl stop pod 2>/dev/null || echo "runpodctl not found. Stop manually."
fi

exit $EXIT_CODE
"""

    wrapper_path = os.path.join(project_dir, '.tmux_eval_wrapper.sh')
    with open(wrapper_path, 'w', newline='\n') as f:
        f.write(wrapper_script)
    os.chmod(wrapper_path, 0o755)

    check = subprocess.run(['tmux', 'has-session', '-t', session_name], capture_output=True)
    if check.returncode == 0:
        print(f"tmux session '{session_name}' already exists.")
        print(f"Attach with: tmux attach -t {session_name}")
        print(f"Or kill it first: tmux kill-session -t {session_name}")
        sys.exit(1)

    subprocess.run([
        'tmux', 'new-session', '-d', '-s', session_name, f'bash {shlex.quote(wrapper_path)}'
    ], check=True)

    print(f"Evaluations running in tmux session: {session_name}")
    print(f"  Attach:  tmux attach -t {session_name}")
    print(f"  Detach:  Ctrl+B then D")


def main():
    parser = argparse.ArgumentParser(description='Evaluate all trained models and produce comparison reports')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Explicit list of checkpoint paths. If omitted, scans experiments/trained_models/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dry_run', action='store_true', help='List models without evaluating')
    parser.add_argument('--tmux', type=str, default=None, metavar='SESSION',
                        help='Run in tmux session (survives disconnects, auto-stops RunPod when done)')
    args = parser.parse_args()

    if args.tmux:
        launch_in_tmux(args.tmux, sys.argv)
        return

    # Discover models
    if args.models:
        checkpoints = args.models
    else:
        checkpoints = discover_models(BASE_DIR)

    if not checkpoints:
        print(f"No models found. Train first with run.py or provide --models paths.")
        return

    print(f"Found {len(checkpoints)} model(s) to evaluate:")
    for i, cp in enumerate(checkpoints, 1):
        folder = os.path.basename(os.path.dirname(cp))
        desc = get_description(os.path.dirname(cp))
        print(f"  {i}. {desc:50s} {cp}")

    if args.dry_run:
        return

    # Load test dataset once (shared across all models)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    test_dir = os.path.join('data', 'keypoints', 'test')
    test_dataset = ASLKeypointDataset(test_dir, augment=False)
    num_classes = len(test_dataset.labels)
    print(f"Test set: {len(test_dataset)} samples, {num_classes} classes")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_eval, num_workers=0)

    # Evaluate each model
    user_killed = False
    def handle_signal(signum, frame):
        nonlocal user_killed
        user_killed = True
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    all_results = []
    try:
        for cp in checkpoints:
            results = evaluate_model(cp, test_dataset, test_loader, device)
            all_results.append(results)
    except KeyboardInterrupt:
        user_killed = True
        print("\n\nUser interrupted — saving partial results.")

    if not all_results:
        print("No models were evaluated.")
        if user_killed:
            sys.exit(130)
        sys.exit(1)

    # Write combined outputs to experiments/evaluation/
    os.makedirs(EVAL_DIR, exist_ok=True)

    summary_csv_path = os.path.join(EVAL_DIR, 'eval_summary.csv')
    write_summary_csv(all_results, summary_csv_path)
    print(f"\nSummary CSV: {summary_csv_path}")

    report_path = os.path.join(EVAL_DIR, 'eval_report.md')
    write_report(all_results, report_path)
    print(f"Report:      {report_path}")

    # Final summary to stdout
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':50s} {'Top-1':>7s} {'Top-5':>7s} {'Ratio':>7s}")
    print(f"{'—'*50} {'—'*7} {'—'*7} {'—'*7}")
    for r in all_results:
        print(f"{r['description']:50s} {r['test_top1']:7.4f} {r['test_top5']:7.4f} {r['distance_ratio']:7.4f}")

    best = max(all_results, key=lambda r: r['test_top1'])
    print(f"\nBest: {best['description']} (top-1: {best['test_top1']:.4f})")

    # Sync results to Google Drive
    print(f"\nSyncing evaluation results to Google Drive...")
    rclone_result = subprocess.run(
        ['rclone', '--config', '/workspace/rclone.conf', 'copy',
         'experiments/evaluation/', 'gdrive:adv-com-vis-final/evaluation', '--progress'],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if rclone_result.returncode == 0:
        print("Drive sync complete.")
    else:
        print(f"Drive sync failed (exit code {rclone_result.returncode}). Files are still in experiments/evaluation/ locally.")

    if user_killed:
        sys.exit(130)


if __name__ == '__main__':
    main()
