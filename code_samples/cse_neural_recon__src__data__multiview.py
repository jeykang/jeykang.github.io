"""
Multi-view dataset wrappers.

These wrappers turn a frame-based dataset into a viewset dataset where each item
contains multiple RGB(+depth) views and their poses/intrinsics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset


def _concatdataset_locate(ds: ConcatDataset, index: int) -> Tuple[int, int]:
    """Return (dataset_idx, sample_idx) for a ConcatDataset global index."""
    if index < 0:
        index = len(ds) + index
    if index < 0 or index >= len(ds):
        raise IndexError(index)

    # cumulative_sizes is a list of running totals.
    cum = ds.cumulative_sizes
    dataset_idx = int(np.searchsorted(cum, index, side="right"))
    start = 0 if dataset_idx == 0 else int(cum[dataset_idx - 1])
    sample_idx = index - start
    return dataset_idx, sample_idx


def _concatdataset_global_index(ds: ConcatDataset, dataset_idx: int, sample_idx: int) -> int:
    start = 0 if dataset_idx == 0 else int(ds.cumulative_sizes[dataset_idx - 1])
    return start + sample_idx


@dataclass(frozen=True)
class MultiViewConfig:
    num_views: int = 6
    window: int = 6  # +/- frames around reference (same underlying sequence)
    include_reference: bool = True
    max_tries: int = 20


class CSEViewSetDataset(Dataset):
    """
    Wrap a dataset to return a fixed-size set of views per item.

    Supported base datasets:
    - frame datasets returning {'rgb','depth','pose','K','valid_mask', ...}
    - multi-camera datasets returning {'left': {...}, 'right': {...}, 'timestamp': ...}
      (we treat each camera sample as a separate view).

    Output tensors:
    - views_rgb: (V, 3, H, W)
    - views_depth: (V, H, W) if available
    - views_pose: (V, 4, 4)
    - views_K: (V, 3, 3)
    - views_valid_mask: (V, H, W) if available
    - views_count: scalar int
    """

    def __init__(self, base: Dataset, config: Optional[MultiViewConfig] = None):
        self.base = base
        self.config = config or MultiViewConfig()

    def __len__(self) -> int:
        return len(self.base)

    def _get_within_same_sequence_indices(self, index: int) -> List[int]:
        cfg = self.config
        target_views = int(cfg.num_views)
        if target_views <= 0:
            raise ValueError("num_views must be > 0")

        if isinstance(self.base, ConcatDataset):
            ds_i, local_i = _concatdataset_locate(self.base, index)
            sub = self.base.datasets[ds_i]
            local_len = len(sub)
            min_i = max(0, local_i - int(cfg.window))
            max_i = min(local_len - 1, local_i + int(cfg.window))

            candidates = list(range(min_i, max_i + 1))
            if cfg.include_reference and local_i not in candidates:
                candidates.append(local_i)

            # Sample without replacement; fall back to replacement if needed.
            if len(candidates) >= target_views:
                chosen_local = list(np.random.choice(candidates, size=target_views, replace=False))
            else:
                chosen_local = list(np.random.choice(candidates, size=target_views, replace=True))
            return [_concatdataset_global_index(self.base, ds_i, int(i)) for i in chosen_local]

        # Non-concat dataset: just use global indices.
        n = len(self.base)
        min_i = max(0, index - int(cfg.window))
        max_i = min(n - 1, index + int(cfg.window))
        candidates = list(range(min_i, max_i + 1))
        if cfg.include_reference and index not in candidates:
            candidates.append(index)
        if len(candidates) >= target_views:
            return list(np.random.choice(candidates, size=target_views, replace=False))
        return list(np.random.choice(candidates, size=target_views, replace=True))

    @staticmethod
    def _flatten_sample_to_views(sample: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        # Multi-camera dataset returns a dict of cameras + timestamp.
        if 'cameras' in sample or 'timestamp' in sample and any(isinstance(v, dict) for k, v in sample.items()):
            views: List[Dict[str, torch.Tensor]] = []
            for k, v in sample.items():
                if k == 'timestamp':
                    continue
                if isinstance(v, dict) and 'rgb' in v and 'pose' in v and 'K' in v:
                    views.append(v)
            return views

        # Frame dataset.
        if 'rgb' in sample and 'pose' in sample and 'K' in sample:
            return [sample]

        raise ValueError("Unsupported sample structure for viewset wrapping")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        indices = self._get_within_same_sequence_indices(index)

        view_samples: List[Dict[str, torch.Tensor]] = []
        for idx in indices:
            sample = self.base[int(idx)]
            view_samples.extend(self._flatten_sample_to_views(sample))

        # Truncate/pad to exact V (after flattening, e.g. stereo may double count).
        V = int(self.config.num_views)
        if len(view_samples) >= V:
            view_samples = view_samples[:V]
        else:
            pad = int(V - len(view_samples))
            if len(view_samples) == 0:
                raise RuntimeError("No valid views produced by wrapper")
            view_samples = view_samples + [view_samples[np.random.randint(len(view_samples))] for _ in range(pad)]

        rgbs = torch.stack([v['rgb'] for v in view_samples], dim=0)  # (V, 3, H, W)
        poses = torch.stack([v['pose'] for v in view_samples], dim=0)  # (V, 4, 4)
        Ks = torch.stack([v['K'] for v in view_samples], dim=0)  # (V, 3, 3)

        out: Dict[str, torch.Tensor] = {
            'views_rgb': rgbs,
            'views_pose': poses,
            'views_K': Ks,
            'views_count': torch.tensor(V, dtype=torch.int64),
        }

        if all('depth' in v for v in view_samples):
            depths = torch.stack([v['depth'] for v in view_samples], dim=0)  # (V, H, W)
            out['views_depth'] = depths
            if all('valid_mask' in v for v in view_samples):
                masks = torch.stack([v['valid_mask'] for v in view_samples], dim=0)  # (V, H, W)
                out['views_valid_mask'] = masks

        return out

