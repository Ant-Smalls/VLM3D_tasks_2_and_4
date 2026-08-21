# analyze_attention.py
# purpose: export gated-attention slice weights for a target class (default Lung nodule)
#          and summarize nodule+/− localization stats
# author: Anthony Smaldore
#

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
from modules.transforms import get_valid_transforms, window_set_from_checkpoint
from modules.multiabnormality_classification_model import (
    MultiAbnormalityClassifier,
    CLASS_NAMES,
)

logger = logging.getLogger(__name__)

POS_COLOR = '#2E86DE'
NEG_COLOR = '#EE5A6F'


def parse_args():
    """
    Parses CLI arguments for the slice-attention analysis script.
    
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
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
        '--window_set',
        type=str,
        default=None,
        help='Override window set for all checkpoints. Default: read each checkpoint',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='cuda recommended on Sonic; cpu for smoke tests only',
    )
    return parser.parse_args()


def parse_checkpoints(raw: List[str]) -> Dict[str, str]:
    """
    Parses named checkpoint specs of the form "name=path".
    
    Args:
        raw (list): CLI strings, each "name=path".
    
    Returns:
        dict: Mapping from checkpoint name to checkpoint path.
    
    Raises:
        ValueError: If a spec is missing "=" or has an empty name or path.
    """
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
    """
    Resolves an abnormality class name to its index in CLASS_NAMES.
    
    Args:
        class_name (str): Abnormality class name (case-insensitive).
    
    Returns:
        tuple: (class_index, canonical_class_name) from CLASS_NAMES.
    
    Raises:
        ValueError: If the name is not in CLASS_NAMES.
    """
    name_to_idx = {n.lower(): i for i, n in enumerate(CLASS_NAMES)}
    key = class_name.lower()
    if key not in name_to_idx:
        raise ValueError(
            f"Unknown class '{class_name}'. Valid names: {CLASS_NAMES}"
        )
    idx = name_to_idx[key]
    return idx, CLASS_NAMES[idx]


def class_slug(name: str) -> str:
    """
    Builds a filesystem-safe slug from an abnormality class name.
    
    Args:
        name (str): Abnormality class name.
    
    Returns:
        str: Lowercase slug with non-alphanumeric runs replaced by "_".
    """
    slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    return slug or 'class'


def patient_id_from_path(image_path: str) -> str:
    """
    Reads the patient id from a CT-RATE manifest image path.
    
    Args:
        image_path (str): Volume path of the form .../<patient_id>/<volume_stem>.
    
    Returns:
        str: The patient id (parent directory), or the path basename if there is no parent.
    """
    parts = os.path.normpath(image_path).split(os.sep)
    if len(parts) >= 2:
        return parts[-2]
    return os.path.basename(image_path)


def build_loader(split_data, args, window_set) -> DataLoader:
    """
    Builds a validation DataLoader with the named window set.
    
    Args:
        split_data (list): Split records with "image" and "label" keys.
        args (argparse.Namespace): CLI arguments (spatial_size, pixdim, batch_size, num_workers, device).
        window_set (str): The name of the window set ("default" or "lung_mediastinum_bone").
    
    Returns:
        DataLoader: Unshuffled loader over the split.
    """
    transforms = get_valid_transforms(
        spatial_size=tuple(args.spatial_size),
        pixdim=tuple(args.pixdim),
        window_set=window_set,
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
    """
    Validates the requested device and returns a torch device.
    
    Args:
        requested (str): Device name ("cuda" or "cpu").
    
    Returns:
        torch.device: The requested device.
    
    Raises:
        RuntimeError: If "cuda" is requested but CUDA is not available.
    """
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
    Loads a checkpoint and extracts per-volume slice attention and labels.
    
    Args:
        checkpoint_path (str): Path to a .ckpt file.
        loader (DataLoader): Batched volumes and labels.
        device (torch.device): Device for inference.
    
    Returns:
        tuple: (attention, labels) where attention is [N, S, num_classes] and labels is [N, num_classes].
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
    Computes sharpness stats for one volume's slice attention.
    
    Clips weights away from zero before entropy so log is defined, then renormalizes.
    
    Args:
        weights (ndarray): Softmax-normalized slice weights of shape [S].
    
    Returns:
        dict: Entropy, perplexity, top-k mass, peak slice, and slice count.
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
    """
    Selects all class-positive indices and a matched-size random negative subset.
    
    Args:
        labels_class (ndarray): Per-volume labels for the target class.
        seed (int): RNG seed for negative sampling.
    
    Returns:
        tuple: (pos_idx, neg_idx). Negatives are downsampled to match positives when there are more negatives.
    """
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
    """
    Builds a per-volume slice-attention stats table for class-positive and class-negative groups.
    
    Args:
        ckpt_name (str): Checkpoint name written into the table.
        class_name (str): Canonical abnormality class name.
        patient_ids (list): Patient id per split row.
        attention_class (ndarray): Slice attention of shape [N, S] for the target class.
        labels_class (ndarray): Per-volume labels for the target class.
        pos_idx (ndarray): Class-positive volume indices.
        neg_idx (ndarray): Class-negative volume indices.
    
    Returns:
        DataFrame: One row per selected volume with entropy, top-k mass, and peak slice.
    """
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
    """
    Aggregates mean and std of slice-attention stats by checkpoint and group.
    
    Args:
        df (DataFrame): Per-volume stats from build_volume_rows.
    
    Returns:
        DataFrame: One row per (checkpoint, group) with mean and std of each metric.
    """
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
    """
    Writes a mean and standard-deviation slice-attention curve for class-positive and class-negative volumes.
    
    Args:
        attention_class (ndarray): Slice attention of shape [N, S] for the target class.
        pos_idx (ndarray): Class-positive volume indices.
        neg_idx (ndarray): Class-negative volume indices.
        title (str): Plot title.
        out_path (str): Output PNG path.
    """
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
    """
    Writes a per-volume slice-attention bar plot with the peak slice marked.
    
    Args:
        weights (ndarray): Slice attention of shape [S] for one volume.
        patient_id (str): Patient id used in the title and filename.
        group (str): Group label ("pos" or "neg").
        stats (dict): Stats from attention_stats_1d, including "peak_slice".
        out_path (str): Output PNG path.
    """
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
    """
    Writes grouped bars of mean entropy, top-k mass, and perplexity for one checkpoint.
    
    Skips the plot if a group is missing.
    
    Args:
        summary_df (DataFrame): Group summary from summarize_groups.
        ckpt_name (str): Checkpoint name to plot.
        out_path (str): Output PNG path.
    """
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
    """
    Returns the path of the per-checkpoint slice-attention cache file.
    
    Args:
        output_subdir (str): Checkpoint output directory.
    
    Returns:
        str: Path to "attention_cache.npz" under that directory.
    """
    return os.path.join(output_subdir, 'attention_cache.npz')


def load_or_extract(
    ckpt_name: str,
    ckpt_path: str,
    loader: DataLoader,
    device: torch.device,
    output_subdir: str,
    use_cache: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads cached slice attention, or extracts it from the checkpoint and optionally caches it.
    
    Args:
        ckpt_name (str): Checkpoint name.
        ckpt_path (str): Path to a .ckpt file.
        loader (DataLoader): Batched volumes and labels.
        device (torch.device): Device for inference.
        output_subdir (str): Checkpoint output directory for the cache file.
        use_cache (bool): If True, read or write "attention_cache.npz".
    
    Returns:
        tuple: (attention, labels) where attention is [N, S, num_classes] and labels is [N, num_classes].
    """
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
    """
    Writes per-checkpoint CSVs and slice-attention plots for the target class.
    
    Args:
        ckpt_name (str): Checkpoint name used as the output subdirectory.
        class_name (str): Canonical abnormality class name.
        class_idx (int): Index of the target class in CLASS_NAMES.
        patient_ids (list): Patient id per split row.
        attention (ndarray): Slice attention of shape [N, S, num_classes].
        labels (ndarray): Labels of shape [N, num_classes].
        args (argparse.Namespace): CLI arguments (output_dir, seed, max_plot_pos, max_plot_neg).
    
    Returns:
        DataFrame: Per-volume stats for the selected class-positive and class-negative cases.
    """
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
    """
    Writes a cross-checkpoint summary table and a class-positive sharpness comparison plot.
    
    Does nothing if fewer than two checkpoints are present.
    
    Args:
        all_volume_dfs (list): Per-checkpoint DataFrames from write_checkpoint_outputs.
        class_name (str): Canonical abnormality class name.
        output_dir (str): Root output directory.
    """
    if len(all_volume_dfs) < 2:
        return
    merged = pd.concat(all_volume_dfs, ignore_index=True)
    summary = summarize_groups(merged)
    path = os.path.join(output_dir, f'cross_checkpoint_summary_{class_slug(class_name)}.csv')
    summary.to_csv(path, index=False)

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
    """
    Runs slice-attention analysis for each named checkpoint on the chosen split.
    
    Raises:
        ValueError: If a checkpoint spec is invalid or the class name is unknown.
        FileNotFoundError: If the split JSON or a checkpoint path does not exist.
        RuntimeError: If "cuda" is requested but unavailable, or attention row count does not match the split.
    """
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

    volume_dfs = []
    for ckpt_name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
        window_set = window_set_from_checkpoint(
            ckpt_path, override=args.window_set
        )
        logger.info(f'{ckpt_name}: window_set={window_set}')
        loader = build_loader(split_data, args, window_set)
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
