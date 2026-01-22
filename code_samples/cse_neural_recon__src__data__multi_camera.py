"""
Multi-Camera Synchronization and Camera Rig utilities.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d


@dataclass
class CameraInfo:
    """Camera information container."""
    name: str
    side: str  # 'left', 'right', 'front', etc.
    intrinsics: np.ndarray  # 3x3 K matrix
    width: int
    height: int
    extrinsic: np.ndarray  # 4x4 transformation from body to camera
    distortion: Optional[np.ndarray] = None
    model: str = 'pinhole'


class CameraRig:
    """
    Multi-camera rig configuration for MobileX Poles robot.
    
    Manages 6x e-Con AR0234CS cameras arranged around the robot.
    Handles extrinsic calibration and coordinate transformations.
    
    Args:
        config: Dictionary containing camera configurations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.cameras: Dict[str, CameraInfo] = {}
        
        if config is None:
            # Default stereo pair configuration
            self._setup_default_stereo()
        else:
            self._load_config(config)
            
    def _setup_default_stereo(self):
        """Setup default stereo camera configuration."""
        # Default intrinsics for CSE dataset
        K = np.array([
            [610.83, 0, 640],
            [0, 611.78, 360],
            [0, 0, 1]
        ])
        
        # Left camera (reference frame)
        left_extrinsic = np.eye(4)
        self.cameras['left'] = CameraInfo(
            name='left',
            side='left',
            intrinsics=K.copy(),
            width=1280,
            height=720,
            extrinsic=left_extrinsic,
        )
        
        # Right camera (stereo baseline ~12cm)
        right_extrinsic = np.eye(4)
        right_extrinsic[0, 3] = 0.12  # Baseline in x direction
        self.cameras['right'] = CameraInfo(
            name='right',
            side='right',
            intrinsics=K.copy(),
            width=1280,
            height=720,
            extrinsic=right_extrinsic,
        )
        
    def _load_config(self, config: Dict):
        """Load camera configuration from dict."""
        for cam_name, cam_config in config.get('cameras', {}).items():
            self.cameras[cam_name] = CameraInfo(
                name=cam_name,
                side=cam_config.get('side', cam_name),
                intrinsics=np.array(cam_config['intrinsics']).reshape(3, 3),
                width=cam_config['width'],
                height=cam_config['height'],
                extrinsic=np.array(cam_config.get('extrinsic', np.eye(4))).reshape(4, 4),
                distortion=np.array(cam_config.get('distortion', [])),
                model=cam_config.get('model', 'pinhole'),
            )
            
    def add_camera(self, camera: CameraInfo):
        """Add a camera to the rig."""
        self.cameras[camera.name] = camera
        
    def get_camera(self, name: str) -> CameraInfo:
        """Get camera by name."""
        return self.cameras[name]
        
    def get_all_cameras(self) -> List[CameraInfo]:
        """Get all cameras in the rig."""
        return list(self.cameras.values())
        
    def transform_to_body(
        self,
        points_cam: torch.Tensor,
        camera_name: str
    ) -> torch.Tensor:
        """
        Transform points from camera frame to body frame.
        
        Args:
            points_cam: (N, 3) points in camera coordinates
            camera_name: Name of the camera
            
        Returns:
            points_body: (N, 3) points in body coordinates
        """
        cam = self.cameras[camera_name]
        extrinsic = torch.from_numpy(cam.extrinsic).float().to(points_cam.device)
        
        # Inverse of camera extrinsic gives body-to-camera transform
        # We want camera-to-body
        R_cb = extrinsic[:3, :3].T
        t_cb = -R_cb @ extrinsic[:3, 3]
        
        points_body = points_cam @ R_cb.T + t_cb
        return points_body
        
    def transform_to_world(
        self,
        points_cam: torch.Tensor,
        camera_name: str,
        body_pose: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform points from camera frame to world frame.
        
        Args:
            points_cam: (N, 3) points in camera coordinates
            camera_name: Name of the camera
            body_pose: (4, 4) body-to-world transformation
            
        Returns:
            points_world: (N, 3) points in world coordinates
        """
        # First transform to body frame
        points_body = self.transform_to_body(points_cam, camera_name)
        
        # Then transform to world frame
        R_wb = body_pose[:3, :3]
        t_wb = body_pose[:3, 3]
        
        points_world = points_body @ R_wb.T + t_wb
        return points_world


class MultiCameraSynchronizer:
    """
    Synchronize data streams from multiple cameras.
    
    Handles temporal alignment of RGB, depth, and pose data
    across multiple cameras with different capture rates.
    
    Args:
        tolerance: Maximum time difference for synchronization (seconds)
        interpolate: Whether to interpolate poses to exact timestamps
        master_source: Which data source to use as timing reference
    """
    
    def __init__(
        self,
        tolerance: float = 0.05,
        interpolate: bool = True,
        master_source: str = 'pose'
    ):
        self.tolerance = tolerance
        self.interpolate = interpolate
        self.master_source = master_source
        
        self.streams: Dict[str, Dict] = {}
        
    def add_stream(
        self,
        name: str,
        timestamps: np.ndarray,
        data_paths: Optional[List[str]] = None,
        data: Optional[np.ndarray] = None
    ):
        """
        Add a data stream to synchronize.
        
        Args:
            name: Stream identifier
            timestamps: Array of timestamps
            data_paths: List of file paths (for lazy loading)
            data: Actual data array (for poses, etc.)
        """
        self.streams[name] = {
            'timestamps': np.array(timestamps),
            'paths': data_paths,
            'data': data,
        }
        
    def synchronize(self) -> List[Dict]:
        """
        Synchronize all streams.
        
        Returns:
            List of synchronized frame dictionaries, each containing
            indices or interpolated values for each stream.
        """
        if not self.streams:
            return []
            
        # Get master timestamps
        master = self.streams.get(self.master_source)
        if master is None:
            master = list(self.streams.values())[0]
        master_timestamps = master['timestamps']
        
        synchronized = []
        
        for master_ts in master_timestamps:
            frame = {'timestamp': master_ts}
            valid = True
            
            for stream_name, stream in self.streams.items():
                result = self._find_nearest(
                    master_ts,
                    stream['timestamps'],
                    stream.get('data'),
                    stream_name
                )
                
                if result is None:
                    valid = False
                    break
                    
                frame[stream_name] = result
                
            if valid:
                synchronized.append(frame)
                
        return synchronized
        
    def _find_nearest(
        self,
        target_ts: float,
        timestamps: np.ndarray,
        data: Optional[np.ndarray],
        stream_name: str
    ) -> Optional[Dict]:
        """Find nearest sample or interpolate."""
        idx = np.searchsorted(timestamps, target_ts)
        idx = np.clip(idx, 0, len(timestamps) - 1)
        
        # Check both neighbors
        candidates = [idx]
        if idx > 0:
            candidates.append(idx - 1)
            
        best_idx = min(candidates, key=lambda i: abs(timestamps[i] - target_ts))
        time_diff = abs(timestamps[best_idx] - target_ts)
        
        if time_diff > self.tolerance:
            return None
            
        result = {
            'idx': best_idx,
            'timestamp': timestamps[best_idx],
            'time_diff': time_diff,
        }
        
        # Interpolate poses if requested
        if self.interpolate and data is not None and stream_name == 'pose':
            result['value'] = self._interpolate_pose(
                target_ts, timestamps, data
            )
        elif data is not None:
            result['value'] = data[best_idx]
            
        return result
        
    def _interpolate_pose(
        self,
        target_ts: float,
        timestamps: np.ndarray,
        poses: np.ndarray
    ) -> np.ndarray:
        """Interpolate pose using SLERP for rotation."""
        idx = np.searchsorted(timestamps, target_ts)
        
        if idx == 0:
            return poses[0]
        if idx >= len(timestamps):
            return poses[-1]
            
        # Interpolation factor
        t0, t1 = timestamps[idx-1], timestamps[idx]
        alpha = (target_ts - t0) / (t1 - t0 + 1e-10)
        alpha = np.clip(alpha, 0, 1)
        
        pose0, pose1 = poses[idx-1], poses[idx]
        
        # Linear interpolation for translation
        t_interp = (1 - alpha) * pose0[:3, 3] + alpha * pose1[:3, 3]
        
        # SLERP for rotation
        r0 = R.from_matrix(pose0[:3, :3])
        r1 = R.from_matrix(pose1[:3, :3])
        
        from scipy.spatial.transform import Slerp
        slerp = Slerp([0, 1], R.concatenate([r0, r1]))
        r_interp = slerp(alpha)
        
        # Build interpolated pose
        pose_interp = np.eye(4)
        pose_interp[:3, :3] = r_interp.as_matrix()
        pose_interp[:3, 3] = t_interp
        
        return pose_interp


def merge_point_clouds(
    clouds: List[torch.Tensor],
    poses: List[torch.Tensor],
    camera_rig: CameraRig,
    camera_names: List[str],
    voxel_size: Optional[float] = None
) -> torch.Tensor:
    """
    Merge point clouds from multiple cameras into world frame.
    
    Args:
        clouds: List of (N_i, 3) point clouds in camera coordinates
        poses: List of (4, 4) body poses
        camera_rig: Camera rig configuration
        camera_names: Camera name for each cloud
        voxel_size: Optional voxel size for downsampling
        
    Returns:
        merged: (M, 3) merged point cloud in world coordinates
    """
    world_clouds = []
    
    for cloud, pose, cam_name in zip(clouds, poses, camera_names):
        if cloud.numel() == 0:
            continue
            
        # Transform to world coordinates
        world_pts = camera_rig.transform_to_world(cloud, cam_name, pose)
        world_clouds.append(world_pts)
        
    if not world_clouds:
        return torch.empty(0, 3)
        
    merged = torch.cat(world_clouds, dim=0)
    
    # Optional voxel downsampling
    if voxel_size is not None:
        merged = voxel_downsample(merged, voxel_size)
        
    return merged


def voxel_downsample(
    points: torch.Tensor,
    voxel_size: float
) -> torch.Tensor:
    """
    Voxel grid downsampling for point clouds.
    
    Args:
        points: (N, 3) point cloud
        voxel_size: Size of voxel grid
        
    Returns:
        downsampled: (M, 3) downsampled point cloud
    """
    if points.numel() == 0:
        return points
        
    # Quantize to voxel grid
    coords = (points / voxel_size).floor().long()
    
    # Compute unique voxel indices
    # Use a simple hash: x + y*max_x + z*max_x*max_y
    min_coords = coords.min(dim=0)[0]
    coords = coords - min_coords
    
    max_coords = coords.max(dim=0)[0] + 1
    voxel_idx = (coords[:, 0] + 
                 coords[:, 1] * max_coords[0] + 
                 coords[:, 2] * max_coords[0] * max_coords[1])
    
    # Get unique voxels and average points within each
    unique_idx, inverse = torch.unique(voxel_idx, return_inverse=True)
    
    # Compute centroids
    centroids = torch.zeros(len(unique_idx), 3, device=points.device)
    counts = torch.zeros(len(unique_idx), device=points.device)
    
    centroids.index_add_(0, inverse, points)
    counts.index_add_(0, inverse, torch.ones(len(points), device=points.device))
    
    centroids = centroids / counts.unsqueeze(1)
    
    return centroids
