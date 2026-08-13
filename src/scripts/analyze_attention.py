# analyze_attention.py
# purpose: export gated-attention slice weights for a target class (default Lung nodule)
#          and summarize nodule+/− localization stats for paper Block-2 diagnostics
# author: Anthony Smaldore
#
# Sonic example (1 GPU, frozen-600 + LoRA-600):
#
#   python3 src/scripts/analyze_attention.py \
#     --checkpoints frozen600=src/outputs/checkpoints/best-epoch=13-val_mean_auc=0.6951.ckpt \
#                   lora600=src/outputs_lora/checkpoints/best-epoch=09-val_mean_auc=0.7071.ckpt \
#     --data_dir $DATA_DIR \
#     --class_name "Lung nodule" \
#     --output_dir src/outputs/attention_analysis \
#     --batch_size 4 \
#     --device cuda

import os
import sys
import json
import argparse
import logging
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from monai.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.transforms import get_valid_transforms
from modules.multiabnormality_classification_model import (
    MultiAbnormalityClassifier,
    CLASS_NAMES,
)

logger = logging.getLogger(__name__)

POS_COLOR = '#2E86DE'
NEG_COLOR = '#EE5A6F'


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Analyze per-slice gated-attention weights for a target abnormality '
            'class (nodule+/− case study).'
        )
    )
    parser.add_argument(
        '--checkpoints',
        nargs='+',
        required=True,
        help='Named checkpoints as name=path (e.g. frozen600=/path/to.ckpt)',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Patient data directory (same as train.py / test.py)',
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        help='Split JSON (default: <data_dir>/subset_manifest/test_split.json)',
    )
    parser.add_argument(
        '--class_name',
        type=str,
        default='Lung nodule',
        help='Abnormality class to analyze (default: Lung nodule)',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Root output dir; writes <name>/ subdirs per checkpoint',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='RNG seed for matching nodule− subset size to nodule+',
    )
    parser.add_argument(
        '--max_plot_pos',
        type=int,
        default=8,
        help='Max individual nodule+ attention plots per checkpoint',
    )
    parser.add_argument(
        '--max_plot_neg',
        type=int,
        default=8,
        help='Max individual nodule− attention plots per checkpoint',
    )
    parser.add_argument(
        '--cache_attention',
        action='store_true',
        help='Save/load per-checkpoint .npz attention caches under each subdir',
    )
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--spatial_size', type=int, nargs=3, default=(96, 224, 224))
    parser.add_argument('--pixdim', type=float, nargs=3, default=(1.5, 1.5, 1.5))
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='cuda recommended on Sonic; cpu for smoke tests only',
    )
    return parser.parse_args()


def parse_checkpoints(raw: List[str]) -> Dict[str, str]:
    checkpoints = {}
    for item in raw:
        if '=' not in item:
            raise ValueError(
                f"Checkpoint '{item}' must be name=path "
                f"(e.g. frozen600=/path/to.ckpt)"
            )
        name, path = item.split('=', 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid checkpoint spec: '{item}'")
        checkpoints[name] = path
    return checkpoints


def resolve_class_index(class_name: str) -> Tuple[int, str]:
    name_to_idx = {n.lower(): i for i, n in enumerate(CLASS_NAMES)}
    key = class_name.lower()
    if key not in name_to_idx:
        raise ValueError(
            f"Unknown class '{class_name}'. Valid names: {CLASS_NAMES}"
        )
    idx = name_to_idx[key]
    return idx, CLASS_NAMES[idx]


def class_slug(name: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    return slug or 'class'


def patient_id_from_path(image_path: str) -> str:
    # manifest paths look like .../<patient_id>/<volume_stem>
    parts = os.path.normpath(image_path).split(os.sep)
    if len(parts) >= 2:
        return parts[-2]
    return os.path.basename(image_path)


def build_loader(split_data, args) -> DataLoader:
    transforms = get_valid_transforms(
        spatial_size=tuple(args.spatial_size),
        pixdim=tuple(args.pixdim),
    )
    ds = Dataset(data=split_data, transform=transforms)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
    )


def resolve_device(requested: str) -> torch.device:
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA requested but not available. Pass --device cpu for smoke tests.'
            )
        return torch.device('cuda')
    return torch.device('cpu')


def extract_attention_for_checkpoint(
    checkpoint_path: str,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        attention: [N, S, num_classes]
        labels: [N, num_classes]
    """
    logger.info(f'Loading checkpoint: {checkpoint_path}')
    model = MultiAbnormalityClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)

    a_batches = []
    label_batches = []
    for batch in tqdm(loader, desc=f'Attention {os.path.basename(checkpoint_path)}'):
        images = batch['image'].to(device)
        labels = batch['label']
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.asarray(labels)
        a = model.extract_attention(images)
        a_batches.append(a.detach().cpu().numpy())
        label_batches.append(labels_np)

    a_all = np.concatenate(a_batches, axis=0)
    labels_all = np.concatenate(label_batches, axis=0)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return a_all, labels_all


def attention_stats_1d(weights: np.ndarray) -> Dict[str, float]:
    """
    Stats for one volume's attention over slices (already softmax-normalized).
    """
    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 1e-12, 1.0)
    w = w / w.sum()
    entropy = float(-np.sum(w * np.log(w)))
    order = np.argsort(w)[::-1]
    top1 = float(w[order[0]])
    top5 = float(w[order[:5]].sum())
    top10 = float(w[order[:10]].sum())
    return {
        'entropy': entropy,
        'perplexity': float(np.exp(entropy)),
        'top1_mass': top1,
        'top5_mass': top5,
        'top10_mass': top10,
        'peak_slice': int(order[0]),
        'n_slices': int(w.shape[0]),
    }


def select_case_indices(
    labels_class: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """All positives; matched-size random negatives (or all negatives if fewer)."""
    pos_idx = np.where(labels_class >= 0.5)[0]
    neg_idx = np.where(labels_class < 0.5)[0]
    rng = np.random.default_rng(seed)
    n_pos = len(pos_idx)
    if len(neg_idx) > n_pos:
        neg_idx = rng.choice(neg_idx, size=n_pos, replace=False)
        neg_idx = np.sort(neg_idx)
    return pos_idx, neg_idx


def build_volume_rows(
    ckpt_name: str,
    class_name: str,
    patient_ids: List[str],
    attention_class: np.ndarray,
    labels_class: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for group, indices in (('pos', pos_idx), ('neg', neg_idx)):
        for i in indices:
            stats = attention_stats_1d(attention_class[i])
            rows.append({
                'checkpoint': ckpt_name,
                'class': class_name,
                'group': group,
                'patient_id': patient_ids[i],
                'split_index': int(i),
                'label': float(labels_class[i]),
                **stats,
            })
    return pd.DataFrame(rows)


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        'entropy', 'perplexity', 'top1_mass', 'top5_mass', 'top10_mass', 'peak_slice',
    ]
    rows = []
    for (ckpt, group), sub in df.groupby(['checkpoint', 'group']):
        row = {
            'checkpoint': ckpt,
            'group': group,
            'n': len(sub),
        }
        for m in metrics:
            row[f'{m}_mean'] = float(sub[m].mean())
            row[f'{m}_std'] = float(sub[m].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_mean_attention_curves(
    attention_class: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    title: str,
    out_path: str,
):
    pos = attention_class[pos_idx]
    neg = attention_class[neg_idx]
    s = np.arange(attention_class.shape[1])

    fig, ax = plt.subplots(figsize=(10, 4))
    if len(pos):
        mu, sd = pos.mean(axis=0), pos.std(axis=0)
        ax.plot(s, mu, color=POS_COLOR, label=f'nodule+ (n={len(pos)})')
        ax.fill_between(s, mu - sd, mu + sd, color=POS_COLOR, alpha=0.2)
    if len(neg):
        mu, sd = neg.mean(axis=0), neg.std(axis=0)
        ax.plot(s, mu, color=NEG_COLOR, label=f'nodule− (n={len(neg)})')
        ax.fill_between(s, mu - sd, mu + sd, color=NEG_COLOR, alpha=0.2)
    ax.set_xlabel('Slice index (post-transform depth)')
    ax.set_ylabel('Attention weight')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.set_xlim(0, attention_class.shape[1] - 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_individual_attention(
    weights: np.ndarray,
    patient_id: str,
    group: str,
    stats: Dict[str, float],
    out_path: str,
):
    s = np.arange(weights.shape[0])
    color = POS_COLOR if group == 'pos' else NEG_COLOR
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(s, weights, color=color, width=1.0, linewidth=0)
    ax.axvline(stats['peak_slice'], color='black', linestyle='--', linewidth=1)
    ax.set_title(
        f'{patient_id} ({group}) | peak={stats["peak_slice"]} '
        f'top5={stats["top5_mass"]:.2f} H={stats["entropy"]:.2f}'
    )
    ax.set_xlabel('Slice index')
    ax.set_ylabel('Attention')
    ax.set_xlim(-0.5, weights.shape[0] - 0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_group_stat_bars(summary_df: pd.DataFrame, ckpt_name: str, out_path: str):
    sub = summary_df[summary_df['checkpoint'] == ckpt_name]
    if sub.empty:
        return
    metrics = ['entropy_mean', 'top1_mass_mean', 'top5_mass_mean', 'perplexity_mean']
    labels = ['entropy', 'top1 mass', 'top5 mass', 'perplexity']
    pos = sub[sub['group'] == 'pos']
    neg = sub[sub['group'] == 'neg']
    if pos.empty or neg.empty:
        return

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        x - width / 2,
        [float(pos.iloc[0][m]) for m in metrics],
        width,
        label='nodule+',
        color=POS_COLOR,
    )
    ax.bar(
        x + width / 2,
        [float(neg.iloc[0][m]) for m in metrics],
        width,
        label='nodule−',
        color=NEG_COLOR,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f'{ckpt_name}: attention mass / sharpness')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def cache_path(output_subdir: str) -> str:
    return os.path.join(output_subdir, 'attention_cache.npz')


def load_or_extract(
    ckpt_name: str,
    ckpt_path: str,
    loader: DataLoader,
    device: torch.device,
    output_subdir: str,
    use_cache: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    os.makedirs(output_subdir, exist_ok=True)
    path = cache_path(output_subdir)
    if use_cache and os.path.exists(path):
        logger.info(f'Loading attention cache: {path}')
        data = np.load(path)
        return data['attention'], data['labels']

    attention, labels = extract_attention_for_checkpoint(ckpt_path, loader, device)
    if use_cache:
        np.savez_compressed(path, attention=attention, labels=labels)
        logger.info(f'Saved attention cache: {path}')
    return attention, labels


def write_checkpoint_outputs(
    ckpt_name: str,
    class_name: str,
    class_idx: int,
    patient_ids: List[str],
    attention: np.ndarray,
    labels: np.ndarray,
    args,
) -> pd.DataFrame:
    out_dir = os.path.join(args.output_dir, ckpt_name)
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, 'cases')
    os.makedirs(plot_dir, exist_ok=True)

    attention_class = attention[:, :, class_idx]
    labels_class = labels[:, class_idx]
    pos_idx, neg_idx = select_case_indices(labels_class, args.seed)

    logger.info(
        f'{ckpt_name}: class={class_name} nodule+={len(pos_idx)} '
        f'matched nodule−={len(neg_idx)} (from {int((labels_class < 0.5).sum())} negatives)'
    )

    volume_df = build_volume_rows(
        ckpt_name, class_name, patient_ids, attention_class, labels_class, pos_idx, neg_idx
    )
    volume_csv = os.path.join(out_dir, f'volume_stats_{class_slug(class_name)}.csv')
    volume_df.to_csv(volume_csv, index=False)

    summary_df = summarize_groups(volume_df)
    summary_csv = os.path.join(out_dir, f'group_summary_{class_slug(class_name)}.csv')
    summary_df.to_csv(summary_csv, index=False)

    plot_mean_attention_curves(
        attention_class,
        pos_idx,
        neg_idx,
        title=f'{ckpt_name}: mean attention — {class_name}',
        out_path=os.path.join(out_dir, f'mean_attention_{class_slug(class_name)}.png'),
    )
    plot_group_stat_bars(
        summary_df,
        ckpt_name,
        out_path=os.path.join(out_dir, f'group_stats_{class_slug(class_name)}.png'),
    )

    # individual case plots (subset)
    for group, indices, limit in (
        ('pos', pos_idx, args.max_plot_pos),
        ('neg', neg_idx, args.max_plot_neg),
    ):
        for i in indices[:limit]:
            stats = attention_stats_1d(attention_class[i])
            pid = patient_ids[i]
            out_path = os.path.join(
                plot_dir,
                f'{group}_{pid}_peak{stats["peak_slice"]}.png',
            )
            plot_individual_attention(
                attention_class[i], pid, group, stats, out_path
            )

    logger.info(f'{ckpt_name}: wrote CSV/PNG under {out_dir}')
    return volume_df


def write_cross_checkpoint_comparison(
    all_volume_dfs: List[pd.DataFrame],
    class_name: str,
    output_dir: str,
):
    if len(all_volume_dfs) < 2:
        return
    merged = pd.concat(all_volume_dfs, ignore_index=True)
    # pivot mean group stats per checkpoint for a compact comparison table
    summary = summarize_groups(merged)
    path = os.path.join(output_dir, f'cross_checkpoint_summary_{class_slug(class_name)}.csv')
    summary.to_csv(path, index=False)

    # side-by-side entropy / top5 for pos group
    pos = summary[summary['group'] == 'pos'].set_index('checkpoint')
    if len(pos) >= 2:
        metrics = ['entropy_mean', 'top5_mass_mean', 'top1_mass_mean', 'perplexity_mean']
        labels = ['entropy', 'top5 mass', 'top1 mass', 'perplexity']
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(metrics))
        width = 0.8 / len(pos)
        for i, ckpt in enumerate(pos.index):
            ax.bar(
                x + i * width,
                [float(pos.loc[ckpt, m]) for m in metrics],
                width,
                label=str(ckpt),
            )
        ax.set_xticks(x + width * (len(pos) - 1) / 2)
        ax.set_xticklabels(labels)
        ax.set_title(f'nodule+ attention sharpness — {class_name}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f'cross_checkpoint_pos_{class_slug(class_name)}.png'),
            dpi=150,
        )
        plt.close(fig)
    logger.info(f'Wrote cross-checkpoint comparison under {output_dir}')


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoints)
    class_idx, class_name = resolve_class_index(args.class_name)
    device = resolve_device(args.device)

    split_path = args.split or os.path.join(
        args.data_dir, 'subset_manifest', 'test_split.json'
    )
    if not os.path.exists(split_path):
        raise FileNotFoundError(f'Split not found: {split_path}')
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    patient_ids = [patient_id_from_path(item['image']) for item in split_data]
    logger.info(f'Loaded split {split_path} with {len(split_data)} volumes')

    os.makedirs(args.output_dir, exist_ok=True)
    loader = build_loader(split_data, args)

    volume_dfs = []
    for ckpt_name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
        subdir = os.path.join(args.output_dir, ckpt_name)
        attention, labels = load_or_extract(
            ckpt_name, ckpt_path, loader, device, subdir, args.cache_attention
        )
        if attention.shape[0] != len(patient_ids):
            raise RuntimeError(
                f'Attention rows ({attention.shape[0]}) != split size ({len(patient_ids)})'
            )
        volume_dfs.append(
            write_checkpoint_outputs(
                ckpt_name, class_name, class_idx, patient_ids, attention, labels, args
            )
        )

    write_cross_checkpoint_comparison(volume_dfs, class_name, args.output_dir)
    logger.info('Attention analysis complete')


if __name__ == '__main__':
    main()
