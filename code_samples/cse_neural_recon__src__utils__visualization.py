"""
Visualization utilities for point clouds, meshes, and depth maps.
"""

from typing import Optional, List, Tuple, Union
from pathlib import Path
import numpy as np


def visualize_point_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    point_size: float = 1.0,
    title: str = "Point Cloud",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize point cloud using Open3D or matplotlib.
    
    Args:
        points: Point coordinates [N, 3]
        colors: Optional colors [N, 3] (0-1 range)
        normals: Optional normals [N, 3]
        point_size: Size of rendered points
        title: Window title
        save_path: Path to save screenshot
        show: Whether to display interactively
    """
    try:
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
            
        if show:
            o3d.visualization.draw_geometries(
                [pcd],
                window_name=title,
                point_show_normal=(normals is not None)
            )
            
        if save_path:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(save_path)
            vis.destroy_window()
            
    except ImportError:
        # Fallback to matplotlib
        _visualize_point_cloud_matplotlib(
            points, colors, title, save_path, show
        )


def _visualize_point_cloud_matplotlib(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    title: str,
    save_path: Optional[str],
    show: bool
):
    """Matplotlib fallback for point cloud visualization."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample if too many points
    if len(points) > 10000:
        indices = np.random.choice(len(points), 10000, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
            
    if colors is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c=colors, s=1, alpha=0.5)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   s=1, alpha=0.5)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = np.max(points.max(axis=0) - points.min(axis=0)) / 2
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def visualize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray] = None,
    title: str = "Mesh",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize triangle mesh using Open3D.
    
    Args:
        vertices: Vertex positions [V, 3]
        faces: Face indices [F, 3]
        colors: Optional vertex colors [V, 3]
        title: Window title
        save_path: Path to save screenshot
        show: Whether to display interactively
    """
    try:
        import open3d as o3d
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        if colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
        mesh.compute_vertex_normals()
        
        if show:
            o3d.visualization.draw_geometries([mesh], window_name=title)
            
        if save_path:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.add_geometry(mesh)
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(save_path)
            vis.destroy_window()
            
    except ImportError:
        print("Open3D required for mesh visualization")


def visualize_depth(
    depth: np.ndarray,
    colormap: str = 'viridis',
    title: str = "Depth",
    save_path: Optional[str] = None,
    show: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
):
    """
    Visualize depth map.
    
    Args:
        depth: Depth values [H, W]
        colormap: Matplotlib colormap name
        title: Plot title
        save_path: Path to save image
        show: Whether to display
        vmin, vmax: Color range limits
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Handle invalid depths
    valid_mask = (depth > 0) & np.isfinite(depth)
    
    if vmin is None:
        vmin = depth[valid_mask].min() if valid_mask.any() else 0
    if vmax is None:
        vmax = depth[valid_mask].max() if valid_mask.any() else 1
        
    im = ax.imshow(depth, cmap=colormap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Depth')
    ax.set_title(title)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def visualize_sdf_slice(
    sdf_func,
    axis: int = 2,
    slice_position: float = 0.0,
    bounds: Tuple[float, float] = (-1, 1),
    resolution: int = 256,
    title: str = "SDF Slice",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize 2D slice of SDF.
    
    Args:
        sdf_func: Function that takes [N, 3] and returns [N] SDF values
        axis: Axis perpendicular to slice (0=X, 1=Y, 2=Z)
        slice_position: Position along the slice axis
        bounds: Spatial bounds for the slice
        resolution: Grid resolution
        title: Plot title
        save_path: Path to save image
        show: Whether to display
    """
    import matplotlib.pyplot as plt
    
    # Create 2D grid
    u = np.linspace(bounds[0], bounds[1], resolution)
    v = np.linspace(bounds[0], bounds[1], resolution)
    uu, vv = np.meshgrid(u, v)
    
    # Create 3D points
    if axis == 0:  # YZ slice
        points = np.stack([
            np.full_like(uu, slice_position),
            uu, vv
        ], axis=-1)
        xlabel, ylabel = 'Y', 'Z'
    elif axis == 1:  # XZ slice
        points = np.stack([
            uu,
            np.full_like(uu, slice_position),
            vv
        ], axis=-1)
        xlabel, ylabel = 'X', 'Z'
    else:  # XY slice
        points = np.stack([
            uu, vv,
            np.full_like(uu, slice_position)
        ], axis=-1)
        xlabel, ylabel = 'X', 'Y'
        
    points = points.reshape(-1, 3)
    
    # Evaluate SDF
    sdf_values = sdf_func(points)
    sdf_grid = sdf_values.reshape(resolution, resolution)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Show SDF values
    im = ax.imshow(
        sdf_grid,
        extent=[bounds[0], bounds[1], bounds[0], bounds[1]],
        origin='lower',
        cmap='RdBu',
        vmin=-0.1,
        vmax=0.1
    )
    
    # Show zero level set
    ax.contour(
        uu, vv, sdf_grid,
        levels=[0],
        colors='k',
        linewidths=2
    )
    
    plt.colorbar(im, ax=ax, label='SDF')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def create_video_from_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30,
    codec: str = 'mp4v'
):
    """
    Create video from frame images.
    
    Args:
        frames: List of image frames [H, W, 3]
        output_path: Output video path
        fps: Frames per second
        codec: Video codec
    """
    try:
        import cv2
        
        if len(frames) == 0:
            raise ValueError("No frames provided")
            
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for frame in frames:
            # Convert to BGR if needed
            if frame.shape[-1] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
                
            # Ensure uint8
            if frame_bgr.dtype != np.uint8:
                frame_bgr = (frame_bgr * 255).astype(np.uint8)
                
            writer.write(frame_bgr)
            
        writer.release()
        
    except ImportError:
        print("OpenCV required for video creation")


def visualize_camera_trajectory(
    poses: List[np.ndarray],
    title: str = "Camera Trajectory",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize camera trajectory in 3D.
    
    Args:
        poses: List of camera poses [4x4 matrices]
        title: Plot title
        save_path: Path to save image
        show: Whether to display
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    positions = np.array([pose[:3, 3] for pose in poses])
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               c='g', s=100, marker='o', label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               c='r', s=100, marker='x', label='End')
    
    # Draw camera orientations (every Nth frame)
    step = max(1, len(poses) // 20)
    for i in range(0, len(poses), step):
        pose = poses[i]
        pos = pose[:3, 3]
        
        # Draw axes
        scale = 0.1
        for j, color in enumerate(['r', 'g', 'b']):
            direction = pose[:3, j] * scale
            ax.quiver(pos[0], pos[1], pos[2],
                     direction[0], direction[1], direction[2],
                     color=color, arrow_length_ratio=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
