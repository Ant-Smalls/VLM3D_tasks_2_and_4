# visualize_embeddings.py
# purpose: extract class-specific attention-pooled embeddings from checkpoints
#          and save PCA / separation comparison figures
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
from sklearn.decomposition import PCA
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
    Parses CLI arguments for class-embedding visualization.
    
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Visualize class-specific bag embeddings (attention-pooled z) '
            'for one or more MultiAbnormalityClassifier checkpoints.'
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
        '--classes',
        nargs='+',
        default=['all'],
        help='Class names to analyze, or "all" for all 18 (default: all)',
    )
    parser.add_argument(
        '--view',
        choices=['all_classes', 'cross_checkpoint', 'inter_class', 'both'],
        default='both',
        help=(
            'Which figure(s) to write. all_classes and both also write '
            'inter_class (positives-only multi-class PCA) per checkpoint.'
        ),
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory for PNGs, comparison_summary.csv, and optional caches',
    )
    parser.add_argument(
        '--cache_embeddings',
        action='store_true',
        help='Save/load per-checkpoint .npz embedding caches under output_dir',
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


def resolve_class_indices(class_args: List[str]) -> List[Tuple[int, str]]:
    """
    Resolves class-name arguments to (index, canonical_name) pairs from CLASS_NAMES.
    
    Args:
        class_args (list): Class names, or a single "all" to select every class.
    
    Returns:
        list: (class_index, canonical_class_name) pairs, without duplicates.
    
    Raises:
        ValueError: If a name is unknown or no classes are selected.
    """
    if len(class_args) == 1 and class_args[0].lower() == 'all':
        return list(enumerate(CLASS_NAMES))

    name_to_idx = {n.lower(): i for i, n in enumerate(CLASS_NAMES)}
    selected = []
    seen = set()
    for raw in class_args:
        key = raw.lower()
        if key not in name_to_idx:
            raise ValueError(
                f"Unknown class '{raw}'. Valid names: {CLASS_NAMES}"
            )
        idx = name_to_idx[key]
        if idx not in seen:
            selected.append((idx, CLASS_NAMES[idx]))
            seen.add(idx)
    if not selected:
        raise ValueError('No classes selected')
    return selected


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


def extract_embeddings_for_checkpoint(
    checkpoint_path: str,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a checkpoint and extracts per-volume class embeddings and labels.
    
    Args:
        checkpoint_path (str): Path to a .ckpt file.
        loader (DataLoader): Batched volumes and labels.
        device (torch.device): Device for inference.
    
    Returns:
        tuple: (z, labels) where z is [N, num_classes, emb_dim] and labels is [N, num_classes].
    """
    logger.info(f'Loading checkpoint: {checkpoint_path}')
    model = MultiAbnormalityClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)

    z_batches = []
    label_batches = []
    for batch in tqdm(loader, desc=f'Extract {os.path.basename(checkpoint_path)}'):
        images = batch['image'].to(device)
        labels = batch['label']
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.asarray(labels)
        z = model.extract_class_embeddings(images)
        z_batches.append(z.detach().cpu().numpy())
        label_batches.append(labels_np)

    z_all = np.concatenate(z_batches, axis=0)
    labels_all = np.concatenate(label_batches, axis=0)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return z_all, labels_all


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """
    Divides vectors by their L2 norm along an axis.
    
    Args:
        x (ndarray): Input array.
        axis (int, optional): Axis along which to compute the norm. Default is -1.
        eps (float, optional): Minimum norm used to avoid division by zero. Default is 1e-8.
    
    Returns:
        ndarray: Array of the same shape as x, with unit-length vectors along axis.
    """
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norms, eps)


def compute_separation_metrics(
    z_class: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Computes cosine-similarity separation of class-positive embeddings from the positive centroid.
    
    L2-normalizes embeddings, takes the mean of class-positive vectors as the positive centroid, and reports mean cosine similarity of positives and negatives to that centroid.
    
    Args:
        z_class (ndarray): Embeddings of shape [N, D] for one class.
        y (ndarray): Per-volume labels for that class.
    
    Returns:
        dict: mean_pos, mean_neg, separation, n_pos, and n_neg. Includes "sims" when both groups are present.
    """
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return {
            'mean_pos': float('nan'),
            'mean_neg': float('nan'),
            'separation': float('nan'),
            'n_pos': n_pos,
            'n_neg': n_neg,
        }

    z_norm = l2_normalize(z_class, axis=1)
    mu_pos = z_norm[y].mean(axis=0)
    mu_pos = mu_pos / max(np.linalg.norm(mu_pos), 1e-8)
    sims = z_norm @ mu_pos
    mean_pos = float(sims[y].mean())
    mean_neg = float(sims[~y].mean())
    return {
        'mean_pos': mean_pos,
        'mean_neg': mean_neg,
        'separation': mean_pos - mean_neg,
        'n_pos': n_pos,
        'n_neg': n_neg,
        'sims': sims,
    }


def fit_pca_2d(z_class: np.ndarray) -> Tuple[np.ndarray, PCA]:
    """
    Fits a 2-component PCA and returns the projected coordinates.
    
    Args:
        z_class (ndarray): Embeddings of shape [N, D].
    
    Returns:
        tuple: (coords, pca) where coords is [N, 2] and pca is the fitted PCA object.
    """
    pca = PCA(n_components=2)
    coords = pca.fit_transform(z_class)
    return coords, pca


def collect_positive_embeddings(
    z: np.ndarray,
    labels: np.ndarray,
    class_indices: List[Tuple[int, str]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Stacks class-specific embeddings for class-positive volumes only.
    
    A multilabel volume contributes one point per positive class, using that class's attention-pooled embedding.
    
    Args:
        z (ndarray): Embeddings of shape [N, num_classes, D].
        labels (ndarray): Labels of shape [N, num_classes].
        class_indices (list): (class_index, class_name) pairs to include.
    
    Returns:
        tuple: (points, class_ids, names) where points is [M, D], class_ids is [M], and names is the class-name list.
    """
    points = []
    class_ids = []
    names = [name for _, name in class_indices]
    local_id = {c_idx: i for i, (c_idx, _) in enumerate(class_indices)}

    for c_idx, _ in class_indices:
        pos = labels[:, c_idx].astype(bool)
        if not np.any(pos):
            continue
        points.append(z[pos, c_idx, :])
        class_ids.append(np.full(int(pos.sum()), local_id[c_idx], dtype=np.int64))

    if not points:
        return (
            np.zeros((0, z.shape[-1]), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            names,
        )
    return np.concatenate(points, axis=0), np.concatenate(class_ids, axis=0), names


def pairwise_centroid_cosine_distance(
    points: np.ndarray,
    class_ids: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Builds a pairwise centroid cosine-distance matrix (1 minus cosine similarity).
    
    Args:
        points (ndarray): Embeddings of shape [M, D].
        class_ids (ndarray): Class index per point, in [0, n_classes).
        n_classes (int): Number of classes (rows and columns of the matrix).
    
    Returns:
        ndarray: Distance matrix of shape [C, C]. Missing classes are NaN.
    """
    dist = np.full((n_classes, n_classes), np.nan, dtype=np.float64)
    centroids = []
    for i in range(n_classes):
        mask = class_ids == i
        if not np.any(mask):
            centroids.append(None)
            continue
        c = l2_normalize(points[mask], axis=1).mean(axis=0)
        c = c / max(np.linalg.norm(c), 1e-8)
        centroids.append(c)

    for i in range(n_classes):
        if centroids[i] is None:
            continue
        for j in range(n_classes):
            if centroids[j] is None:
                continue
            dist[i, j] = 1.0 - float(np.dot(centroids[i], centroids[j]))
    return dist


def plot_inter_class(
    z: np.ndarray,
    labels: np.ndarray,
    class_indices: List[Tuple[int, str]],
    checkpoint_name: str,
    output_path: str,
    distance_csv_path: str,
) -> None:
    """
    Writes a positives-only PCA scatter and a centroid cosine-distance heatmap.
    
    Skips the plot if fewer than two positive points are present.
    
    Args:
        z (ndarray): Embeddings of shape [N, num_classes, D].
        labels (ndarray): Labels of shape [N, num_classes].
        class_indices (list): (class_index, class_name) pairs to plot.
        checkpoint_name (str): Checkpoint name used in titles and logs.
        output_path (str): Output PNG path.
        distance_csv_path (str): Output CSV path for the distance matrix.
    """
    points, class_ids, names = collect_positive_embeddings(z, labels, class_indices)
    n_classes = len(names)

    if points.shape[0] < 2:
        logger.warning(
            f'inter_class skipped for {checkpoint_name}: '
            f'need >=2 positive points, got {points.shape[0]}'
        )
        return

    coords, pca = fit_pca_2d(points)
    dist = pairwise_centroid_cosine_distance(points, class_ids, n_classes)

    dist_df = pd.DataFrame(dist, index=names, columns=names)
    dist_df.to_csv(distance_csv_path)
    logger.info(f'Wrote {distance_csv_path}')

    cmap = plt.get_cmap('tab20', max(n_classes, 1))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    ax = axes[0]
    for i, name in enumerate(names):
        mask = class_ids == i
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(i)], s=22, alpha=0.75, label=f'{name} (n={int(mask.sum())})',
            edgecolors='none',
        )
        ax.scatter(
            coords[mask, 0].mean(), coords[mask, 1].mean(),
            c=[cmap(i)], s=120, marker='X', edgecolors='black', linewidths=0.6,
        )
    pc1 = pca.explained_variance_ratio_[0]
    pc2 = pca.explained_variance_ratio_[1]
    ax.set_title(
        f'Positive cases by class — {checkpoint_name}\n'
        f'PC1+PC2={pc1 + pc2:.1%}'
    )
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=6, loc='best', frameon=False, markerscale=1.2)

    ax_h = axes[1]
    im = ax_h.imshow(dist, cmap='viridis', vmin=0.0, vmax=1.0, aspect='auto')
    ax_h.set_xticks(range(n_classes))
    ax_h.set_yticks(range(n_classes))
    ax_h.set_xticklabels(names, rotation=90, fontsize=7)
    ax_h.set_yticklabels(names, fontsize=7)
    ax_h.set_title('Centroid cosine distance (1 − cos)')
    fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)

    fig.suptitle(
        f'Inter-class positive separation — {checkpoint_name}',
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Wrote {output_path}')


def plot_all_classes(
    z: np.ndarray,
    labels: np.ndarray,
    class_indices: List[Tuple[int, str]],
    checkpoint_name: str,
    output_path: str,
) -> List[dict]:
    """
    Writes a per-class PCA grid of class-positive vs class-negative embeddings.
    
    Args:
        z (ndarray): Embeddings of shape [N, num_classes, D].
        labels (ndarray): Labels of shape [N, num_classes].
        class_indices (list): (class_index, class_name) pairs to plot.
        checkpoint_name (str): Checkpoint name used in titles and logs.
        output_path (str): Output PNG path.
    
    Returns:
        list: Per-class separation metric rows for the summary table.
    """
    n = len(class_indices)
    ncols = 6 if n > 6 else max(n, 1)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    rows = []
    for ax_i, (c_idx, c_name) in enumerate(class_indices):
        ax = axes[ax_i]
        z_c = z[:, c_idx, :]
        y = labels[:, c_idx]
        metrics = compute_separation_metrics(z_c, y)
        coords, pca = fit_pca_2d(z_c)

        pos = y.astype(bool)
        ax.scatter(
            coords[~pos, 0], coords[~pos, 1],
            c=NEG_COLOR, s=18, alpha=0.7, label='neg', edgecolors='none',
        )
        ax.scatter(
            coords[pos, 0], coords[pos, 1],
            c=POS_COLOR, s=18, alpha=0.7, label='pos', edgecolors='none',
        )
        sep = metrics['separation']
        pc1 = pca.explained_variance_ratio_[0]
        pc2 = pca.explained_variance_ratio_[1]
        ax.set_title(f'{c_name}\nΔ={sep:.3f}' if not np.isnan(sep) else f'{c_name}\nΔ=n/a', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=6, loc='best', frameon=False)

        rows.append({
            'Checkpoint': checkpoint_name,
            'Class': c_name,
            'Mean Pos Sim': metrics['mean_pos'],
            'Mean Neg Sim': metrics['mean_neg'],
            'Separation (Δ)': metrics['separation'],
            'PC1 Variance': pc1,
            'PC2 Variance': pc2,
            'Total Variance': pc1 + pc2,
            'N Pos': metrics['n_pos'],
            'N Neg': metrics['n_neg'],
        })

    for j in range(len(class_indices), len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'All-classes embedding PCA — {checkpoint_name}', fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Wrote {output_path}')
    return rows


def plot_cross_checkpoint(
    embeddings: Dict[str, Dict[str, np.ndarray]],
    class_idx: int,
    class_name: str,
    checkpoint_names: List[str],
    output_path: str,
) -> List[dict]:
    """
    Writes a per-checkpoint PCA and cosine-similarity histogram for one class.
    
    Args:
        embeddings (dict): Mapping from checkpoint name to dicts with "z" and "labels".
        class_idx (int): Index of the target class in CLASS_NAMES.
        class_name (str): Canonical abnormality class name.
        checkpoint_names (list): Checkpoint names in plot order.
        output_path (str): Output PNG path.
    
    Returns:
        list: Per-checkpoint separation metric rows for the summary table.
    """
    n = len(checkpoint_names)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7.0), squeeze=False)

    rows = []
    for col, name in enumerate(checkpoint_names):
        z = embeddings[name]['z'][:, class_idx, :]
        y = embeddings[name]['labels'][:, class_idx]
        metrics = compute_separation_metrics(z, y)
        coords, pca = fit_pca_2d(z)
        pos = y.astype(bool)

        ax_pca = axes[0, col]
        ax_pca.scatter(
            coords[~pos, 0], coords[~pos, 1],
            c=NEG_COLOR, s=28, alpha=0.75, label='neg', edgecolors='none',
        )
        ax_pca.scatter(
            coords[pos, 0], coords[pos, 1],
            c=POS_COLOR, s=28, alpha=0.75, label='pos', edgecolors='none',
        )
        sep = metrics['separation']
        pc1 = pca.explained_variance_ratio_[0]
        pc2 = pca.explained_variance_ratio_[1]
        ax_pca.set_title(
            f'{name}\nΔ={sep:.4f} | PC1+2={pc1 + pc2:.1%}'
            if not np.isnan(sep) else f'{name}\nΔ=n/a'
        )
        ax_pca.set_xlabel('PC1')
        ax_pca.set_ylabel('PC2')
        ax_pca.legend(fontsize=8, frameon=False)

        ax_hist = axes[1, col]
        sims = metrics.get('sims')
        if sims is not None and not np.isnan(sep):
            ax_hist.hist(
                sims[~pos], bins=20, alpha=0.6, color=NEG_COLOR,
                label=f"neg μ={metrics['mean_neg']:.3f}", density=True,
            )
            ax_hist.hist(
                sims[pos], bins=20, alpha=0.6, color=POS_COLOR,
                label=f"pos μ={metrics['mean_pos']:.3f}", density=True,
            )
            ax_hist.axvline(metrics['mean_neg'], color=NEG_COLOR, linestyle='--', linewidth=1.2)
            ax_hist.axvline(metrics['mean_pos'], color=POS_COLOR, linestyle='--', linewidth=1.2)
            ax_hist.set_xlim(-0.5, 1.05)
            ax_hist.legend(fontsize=7, frameon=False)
        else:
            ax_hist.text(0.5, 0.5, 'n/a (missing pos or neg)', ha='center', va='center')
            ax_hist.set_xticks([])
            ax_hist.set_yticks([])
        ax_hist.set_xlabel('Cosine sim to μ_pos')
        ax_hist.set_ylabel('Density')

        rows.append({
            'Checkpoint': name,
            'Class': class_name,
            'Mean Pos Sim': metrics['mean_pos'],
            'Mean Neg Sim': metrics['mean_neg'],
            'Separation (Δ)': metrics['separation'],
            'PC1 Variance': pc1,
            'PC2 Variance': pc2,
            'Total Variance': pc1 + pc2,
            'N Pos': metrics['n_pos'],
            'N Neg': metrics['n_neg'],
        })

    fig.suptitle(f'Cross-checkpoint comparison — {class_name}', fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Wrote {output_path}')
    return rows


def cache_path(output_dir: str, checkpoint_name: str) -> str:
    """
    Returns the path of the per-checkpoint embedding cache file.
    
    Args:
        output_dir (str): Root output directory.
        checkpoint_name (str): Checkpoint name used in the filename.
    
    Returns:
        str: Path to "embeddings_<checkpoint_name>.npz".
    """
    return os.path.join(output_dir, f'embeddings_{checkpoint_name}.npz')


def load_or_extract(
    name: str,
    path: str,
    loader: DataLoader,
    device: torch.device,
    output_dir: str,
    use_cache: bool,
) -> Dict[str, np.ndarray]:
    """
    Loads cached class embeddings, or extracts them from the checkpoint and optionally caches them.
    
    Args:
        name (str): Checkpoint name used in the cache filename.
        path (str): Path to a .ckpt file.
        loader (DataLoader): Batched volumes and labels.
        device (torch.device): Device for inference.
        output_dir (str): Directory for the cache file.
        use_cache (bool): If True, read or write the .npz cache.
    
    Returns:
        dict: Mapping with "z" and "labels" arrays.
    """
    npz_path = cache_path(output_dir, name)
    if use_cache and os.path.exists(npz_path):
        logger.info(f'Loading cached embeddings: {npz_path}')
        data = np.load(npz_path)
        return {'z': data['z'], 'labels': data['labels']}

    z, labels = extract_embeddings_for_checkpoint(path, loader, device)
    if use_cache:
        np.savez_compressed(npz_path, z=z, labels=labels)
        logger.info(f'Cached embeddings: {npz_path}')
    return {'z': z, 'labels': labels}


def build_summary_rows(
    embeddings: Dict[str, Dict[str, np.ndarray]],
    class_indices: List[Tuple[int, str]],
    checkpoint_names: List[str],
) -> List[dict]:
    """
    Builds per-checkpoint, per-class separation metric rows.
    
    Args:
        embeddings (dict): Mapping from checkpoint name to dicts with "z" and "labels".
        class_indices (list): (class_index, class_name) pairs to include.
        checkpoint_names (list): Checkpoint names in row order.
    
    Returns:
        list: Rows for "comparison_summary.csv".
    """
    rows = []
    for name in checkpoint_names:
        z = embeddings[name]['z']
        labels = embeddings[name]['labels']
        for c_idx, c_name in class_indices:
            metrics = compute_separation_metrics(z[:, c_idx, :], labels[:, c_idx])
            _, pca = fit_pca_2d(z[:, c_idx, :])
            pc1 = pca.explained_variance_ratio_[0]
            pc2 = pca.explained_variance_ratio_[1]
            rows.append({
                'Checkpoint': name,
                'Class': c_name,
                'Mean Pos Sim': metrics['mean_pos'],
                'Mean Neg Sim': metrics['mean_neg'],
                'Separation (Δ)': metrics['separation'],
                'PC1 Variance': pc1,
                'PC2 Variance': pc2,
                'Total Variance': pc1 + pc2,
                'N Pos': metrics['n_pos'],
                'N Neg': metrics['n_neg'],
            })
    return rows


def main():
    """
    Extracts class embeddings for each named checkpoint and writes PCA comparison figures.
    
    Raises:
        ValueError: If a checkpoint spec is invalid or a class name is unknown.
        FileNotFoundError: If the split JSON or a checkpoint path does not exist.
        RuntimeError: If "cuda" is requested but CUDA is not available.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoints = parse_checkpoints(args.checkpoints)
    class_indices = resolve_class_indices(args.classes)
    device = resolve_device(args.device)
    logger.info(f'Using device: {device}')
    logger.info(f'Checkpoints: {list(checkpoints.keys())}')
    logger.info(f'Classes ({len(class_indices)}): {[n for _, n in class_indices]}')

    split_path = args.split or os.path.join(
        args.data_dir, 'subset_manifest', 'test_split.json'
    )
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f'Split JSON not found at {split_path}. Pass --split explicitly.'
        )
    for name, path in checkpoints.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f'Checkpoint not found for {name}: {path}')

    with open(split_path, 'r') as f:
        split_data = json.load(f)
    logger.info(f'Loaded split {split_path} ({len(split_data)} studies)')

    checkpoint_names = list(checkpoints.keys())

    embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    for name, path in checkpoints.items():
        window_set = window_set_from_checkpoint(path, override=args.window_set)
        logger.info(f'{name}: window_set={window_set}')
        loader = build_loader(split_data, args, window_set)
        embeddings[name] = load_or_extract(
            name, path, loader, device, args.output_dir, args.cache_embeddings,
        )

    summary_rows: List[dict] = []

    if args.view in ('all_classes', 'both'):
        for name in checkpoint_names:
            out = os.path.join(args.output_dir, f'all_classes_{name}.png')
            plot_all_classes(
                embeddings[name]['z'],
                embeddings[name]['labels'],
                class_indices,
                name,
                out,
            )

    if args.view in ('all_classes', 'inter_class', 'both'):
        for name in checkpoint_names:
            out = os.path.join(args.output_dir, f'inter_class_{name}.png')
            dist_csv = os.path.join(
                args.output_dir, f'inter_class_centroid_distances_{name}.csv'
            )
            plot_inter_class(
                embeddings[name]['z'],
                embeddings[name]['labels'],
                class_indices,
                name,
                out,
                dist_csv,
            )

    if args.view in ('cross_checkpoint', 'both'):
        if len(checkpoint_names) < 2 and args.view == 'cross_checkpoint':
            logger.warning(
                'cross_checkpoint view with a single checkpoint; '
                'figure will still be written with one column.'
            )
        for c_idx, c_name in class_indices:
            out = os.path.join(
                args.output_dir, f'cross_checkpoint_{class_slug(c_name)}.png'
            )
            plot_cross_checkpoint(
                embeddings, c_idx, c_name, checkpoint_names, out,
            )

    summary_rows = build_summary_rows(embeddings, class_indices, checkpoint_names)
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(args.output_dir, 'comparison_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    logger.info(f'Wrote {csv_path}')

    print('\n' + '=' * 100)
    print('EMBEDDING COMPARISON SUMMARY')
    print('=' * 100)
    print(summary_df.to_string(index=False))
    print('=' * 100)

    if not summary_df.empty and summary_df['Separation (Δ)'].notna().any():
        for c_name in summary_df['Class'].unique():
            sub = summary_df[summary_df['Class'] == c_name]
            if sub['Separation (Δ)'].isna().all():
                continue
            best_idx = sub['Separation (Δ)'].idxmax()
            best = sub.loc[best_idx]
            print(
                f"Best separation for {c_name}: {best['Checkpoint']} "
                f"(Δ = {best['Separation (Δ)']:.4f})"
            )


if __name__ == '__main__':
    main()
