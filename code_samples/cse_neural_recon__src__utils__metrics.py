"""
Evaluation metrics for 3D reconstruction.
"""

from typing import Optional, Tuple
import numpy as np


def chamfer_distance(
    pred: np.ndarray,
    target: np.ndarray,
    bidirectional: bool = True
) -> float:
    """
    Compute Chamfer distance between point clouds.
    
    Args:
        pred: Predicted points [N, 3]
        target: Target points [M, 3]
        bidirectional: If True, compute bidirectional CD
        
    Returns:
        Chamfer distance
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        
        # Pred to target
        nn_target = NearestNeighbors(n_neighbors=1)
        nn_target.fit(target)
        distances_pred, _ = nn_target.kneighbors(pred)
        cd_pred = np.mean(distances_pred ** 2)
        
        if not bidirectional:
            return cd_pred
            
        # Target to pred
        nn_pred = NearestNeighbors(n_neighbors=1)
        nn_pred.fit(pred)
        distances_target, _ = nn_pred.kneighbors(target)
        cd_target = np.mean(distances_target ** 2)
        
        return (cd_pred + cd_target) / 2
        
    except ImportError:
        # Brute force computation
        dist_matrix = np.linalg.norm(
            pred[:, None] - target[None, :], axis=-1
        )
        
        cd_pred = np.mean(dist_matrix.min(axis=1) ** 2)
        
        if not bidirectional:
            return cd_pred
            
        cd_target = np.mean(dist_matrix.min(axis=0) ** 2)
        
        return (cd_pred + cd_target) / 2


def point_cloud_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.01
) -> float:
    """
    Compute accuracy: fraction of pred points within threshold of target.
    
    Args:
        pred: Predicted points [N, 3]
        target: Target points [M, 3]
        threshold: Distance threshold
        
    Returns:
        Accuracy (0-1)
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(target)
        distances, _ = nn.kneighbors(pred)
        
        return np.mean(distances.flatten() < threshold)
        
    except ImportError:
        dist_matrix = np.linalg.norm(
            pred[:, None] - target[None, :], axis=-1
        )
        min_dists = dist_matrix.min(axis=1)
        
        return np.mean(min_dists < threshold)


def point_cloud_completeness(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.01
) -> float:
    """
    Compute completeness: fraction of target points within threshold of pred.
    
    Args:
        pred: Predicted points [N, 3]
        target: Target points [M, 3]
        threshold: Distance threshold
        
    Returns:
        Completeness (0-1)
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(pred)
        distances, _ = nn.kneighbors(target)
        
        return np.mean(distances.flatten() < threshold)
        
    except ImportError:
        dist_matrix = np.linalg.norm(
            target[:, None] - pred[None, :], axis=-1
        )
        min_dists = dist_matrix.min(axis=1)
        
        return np.mean(min_dists < threshold)


def f_score(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.01
) -> Tuple[float, float, float]:
    """
    Compute F-score, precision (accuracy), and recall (completeness).
    
    Args:
        pred: Predicted points [N, 3]
        target: Target points [M, 3]
        threshold: Distance threshold
        
    Returns:
        Tuple of (f_score, precision, recall)
    """
    precision = point_cloud_accuracy(pred, target, threshold)
    recall = point_cloud_completeness(pred, target, threshold)
    
    if precision + recall > 0:
        f = 2 * precision * recall / (precision + recall)
    else:
        f = 0.0
        
    return f, precision, recall


def hausdorff_distance(
    pred: np.ndarray,
    target: np.ndarray,
    percentile: Optional[float] = None
) -> float:
    """
    Compute Hausdorff distance between point clouds.
    
    Args:
        pred: Predicted points [N, 3]
        target: Target points [M, 3]
        percentile: If set, compute percentile Hausdorff (e.g., 95)
        
    Returns:
        Hausdorff distance
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        
        # Pred to target
        nn_target = NearestNeighbors(n_neighbors=1)
        nn_target.fit(target)
        distances_pred, _ = nn_target.kneighbors(pred)
        
        # Target to pred
        nn_pred = NearestNeighbors(n_neighbors=1)
        nn_pred.fit(pred)
        distances_target, _ = nn_pred.kneighbors(target)
        
        all_distances = np.concatenate([
            distances_pred.flatten(),
            distances_target.flatten()
        ])
        
        if percentile is not None:
            return np.percentile(all_distances, percentile)
        return np.max(all_distances)
        
    except ImportError:
        dist_matrix = np.linalg.norm(
            pred[:, None] - target[None, :], axis=-1
        )
        
        min_pred_to_target = dist_matrix.min(axis=1)
        min_target_to_pred = dist_matrix.min(axis=0)
        
        all_distances = np.concatenate([min_pred_to_target, min_target_to_pred])
        
        if percentile is not None:
            return np.percentile(all_distances, percentile)
        return np.max(all_distances)


def normal_consistency(
    pred_points: np.ndarray,
    pred_normals: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray
) -> float:
    """
    Compute normal consistency between point clouds.
    
    Args:
        pred_points: Predicted points [N, 3]
        pred_normals: Predicted normals [N, 3]
        target_points: Target points [M, 3]
        target_normals: Target normals [M, 3]
        
    Returns:
        Normal consistency (0-1)
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(target_points)
        _, indices = nn.kneighbors(pred_points)
        
        corresponding_normals = target_normals[indices.flatten()]
        
        # Compute absolute dot product (normals can be flipped)
        dot_products = np.abs(np.sum(
            pred_normals * corresponding_normals, axis=1
        ))
        
        return np.mean(dot_products)
        
    except ImportError:
        # Brute force
        dist_matrix = np.linalg.norm(
            pred_points[:, None] - target_points[None, :], axis=-1
        )
        indices = dist_matrix.argmin(axis=1)
        
        corresponding_normals = target_normals[indices]
        dot_products = np.abs(np.sum(
            pred_normals * corresponding_normals, axis=1
        ))
        
        return np.mean(dot_products)


def earth_movers_distance(
    pred: np.ndarray,
    target: np.ndarray
) -> float:
    """
    Compute Earth Mover's Distance (EMD) between point clouds.
    
    Note: This is computationally expensive for large point clouds.
    Consider using subsampled versions.
    
    Args:
        pred: Predicted points [N, 3]
        target: Target points [N, 3] (must have same count)
        
    Returns:
        EMD value
    """
    assert len(pred) == len(target), "EMD requires same number of points"
    
    try:
        from scipy.optimize import linear_sum_assignment
        
        # Compute cost matrix
        cost_matrix = np.linalg.norm(
            pred[:, None] - target[None, :], axis=-1
        )
        
        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        return cost_matrix[row_ind, col_ind].mean()
        
    except ImportError:
        raise ImportError("scipy required for EMD computation")


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.01,
    pred_normals: Optional[np.ndarray] = None,
    target_normals: Optional[np.ndarray] = None
) -> dict:
    """
    Compute all reconstruction metrics.
    
    Args:
        pred: Predicted points [N, 3]
        target: Target points [M, 3]
        threshold: Distance threshold for F-score
        pred_normals: Optional predicted normals
        target_normals: Optional target normals
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Chamfer distance
    metrics['chamfer_distance'] = chamfer_distance(pred, target)
    
    # F-score
    f, precision, recall = f_score(pred, target, threshold)
    metrics['f_score'] = f
    metrics['precision'] = precision
    metrics['recall'] = recall
    
    # Hausdorff distances
    metrics['hausdorff'] = hausdorff_distance(pred, target)
    metrics['hausdorff_95'] = hausdorff_distance(pred, target, percentile=95)
    
    # Normal consistency
    if pred_normals is not None and target_normals is not None:
        metrics['normal_consistency'] = normal_consistency(
            pred, pred_normals, target, target_normals
        )
        
    return metrics
