"""
File I/O utilities for point clouds, meshes, and images.
"""

from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np


def load_point_cloud(
    path: str,
    load_colors: bool = True,
    load_normals: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load point cloud from file.
    
    Supports PLY, PCD, XYZ, and NPZ formats.
    
    Args:
        path: Path to point cloud file
        load_colors: Whether to load colors if available
        load_normals: Whether to load normals if available
        
    Returns:
        Tuple of (points, colors, normals)
    """
    path = Path(path)
    ext = path.suffix.lower()
    
    if ext == '.ply':
        return _load_ply(path, load_colors, load_normals)
    elif ext == '.pcd':
        return _load_pcd(path)
    elif ext == '.xyz':
        return _load_xyz(path)
    elif ext == '.npz':
        return _load_npz(path, load_colors, load_normals)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def _load_ply(
    path: Path,
    load_colors: bool,
    load_normals: bool
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load PLY file."""
    try:
        import open3d as o3d
        
        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points)
        
        colors = None
        if load_colors and pcd.has_colors():
            colors = np.asarray(pcd.colors)
            
        normals = None
        if load_normals and pcd.has_normals():
            normals = np.asarray(pcd.normals)
            
        return points, colors, normals
        
    except ImportError:
        # Manual PLY parsing
        return _parse_ply_manually(path, load_colors, load_normals)


def _parse_ply_manually(
    path: Path,
    load_colors: bool,
    load_normals: bool
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Manually parse PLY file."""
    with open(path, 'rb') as f:
        # Read header
        header_ended = False
        vertex_count = 0
        properties = []
        
        while not header_ended:
            line = f.readline().decode('utf-8').strip()
            
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                properties.append(parts[-1])
            elif line == 'end_header':
                header_ended = True
                
        # Determine format
        has_colors = any(p in properties for p in ['red', 'r'])
        has_normals = any(p in properties for p in ['nx'])
        
        # Read binary or ascii data
        data = np.loadtxt(f, max_rows=vertex_count)
        
        points = data[:, :3]
        colors = None
        normals = None
        
        idx = 3
        if has_normals and load_normals:
            normals = data[:, idx:idx+3]
            idx += 3
            
        if has_colors and load_colors:
            colors = data[:, idx:idx+3]
            if colors.max() > 1.0:
                colors = colors / 255.0
                
    return points, colors, normals


def _load_pcd(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load PCD file using Open3D."""
    try:
        import open3d as o3d
        
        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        
        return points, colors, normals
        
    except ImportError:
        raise ImportError("Open3D required for PCD files")


def _load_xyz(path: Path) -> Tuple[np.ndarray, None, None]:
    """Load XYZ file (simple text format)."""
    data = np.loadtxt(path)
    return data[:, :3], None, None


def _load_npz(
    path: Path,
    load_colors: bool,
    load_normals: bool
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load NPZ file."""
    data = np.load(path)
    
    points = data['points'] if 'points' in data else data['xyz']
    colors = data.get('colors') if load_colors else None
    normals = data.get('normals') if load_normals else None
    
    return points, colors, normals


def save_point_cloud(
    path: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None
):
    """
    Save point cloud to file.
    
    Args:
        path: Output path
        points: Point coordinates [N, 3]
        colors: Optional colors [N, 3] (0-1 range)
        normals: Optional normals [N, 3]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    ext = path.suffix.lower()
    
    if ext == '.ply':
        _save_ply(path, points, colors, normals)
    elif ext == '.pcd':
        _save_pcd(path, points, colors, normals)
    elif ext == '.xyz':
        np.savetxt(path, points, fmt='%.6f')
    elif ext == '.npz':
        data = {'points': points}
        if colors is not None:
            data['colors'] = colors
        if normals is not None:
            data['normals'] = normals
        np.savez(path, **data)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def _save_ply(
    path: Path,
    points: np.ndarray,
    colors: Optional[np.ndarray],
    normals: Optional[np.ndarray]
):
    """Save PLY file."""
    try:
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
            
        o3d.io.write_point_cloud(str(path), pcd)
        
    except ImportError:
        # Manual PLY writing
        _write_ply_manually(path, points, colors, normals)


def _write_ply_manually(
    path: Path,
    points: np.ndarray,
    colors: Optional[np.ndarray],
    normals: Optional[np.ndarray]
):
    """Manually write PLY file."""
    n = len(points)
    
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if normals is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            
        f.write("end_header\n")
        
        for i in range(n):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            
            if normals is not None:
                line += f" {normals[i, 0]:.6f} {normals[i, 1]:.6f} {normals[i, 2]:.6f}"
                
            if colors is not None:
                c = colors[i]
                if c.max() <= 1.0:
                    c = (c * 255).astype(int)
                line += f" {int(c[0])} {int(c[1])} {int(c[2])}"
                
            f.write(line + "\n")


def _save_pcd(
    path: Path,
    points: np.ndarray,
    colors: Optional[np.ndarray],
    normals: Optional[np.ndarray]
):
    """Save PCD file using Open3D."""
    try:
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
            
        o3d.io.write_point_cloud(str(path), pcd)
        
    except ImportError:
        raise ImportError("Open3D required for PCD files")


def load_mesh(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load triangle mesh from file.
    
    Args:
        path: Path to mesh file (PLY, OBJ, OFF)
        
    Returns:
        Tuple of (vertices, faces, colors)
    """
    path = Path(path)
    ext = path.suffix.lower()
    
    try:
        import open3d as o3d
        
        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
        
        return vertices, faces, colors
        
    except ImportError:
        if ext == '.obj':
            return _load_obj_manually(path)
        raise ImportError("Open3D required for mesh loading")


def _load_obj_manually(path: Path) -> Tuple[np.ndarray, np.ndarray, None]:
    """Manually load OBJ file."""
    vertices = []
    faces = []
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()[1:4]
                vertices.append([float(p) for p in parts])
            elif line.startswith('f '):
                parts = line.split()[1:]
                # Handle different face formats (v, v/vt, v/vt/vn, v//vn)
                face = []
                for p in parts:
                    idx = int(p.split('/')[0]) - 1  # OBJ is 1-indexed
                    face.append(idx)
                if len(face) >= 3:
                    faces.append(face[:3])  # Take first 3 for triangles
                    
    return np.array(vertices), np.array(faces), None


def save_mesh(
    path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray] = None
):
    """
    Save triangle mesh to file.
    
    Args:
        path: Output path
        vertices: Vertex positions [V, 3]
        faces: Face indices [F, 3]
        colors: Optional vertex colors [V, 3]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import open3d as o3d
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        if colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(path), mesh)
        
    except ImportError:
        ext = path.suffix.lower()
        if ext == '.obj':
            _save_obj_manually(path, vertices, faces)
        elif ext == '.ply':
            _save_mesh_ply_manually(path, vertices, faces, colors)
        else:
            raise ImportError("Open3D required for mesh saving")


def _save_obj_manually(path: Path, vertices: np.ndarray, faces: np.ndarray):
    """Manually save OBJ file."""
    with open(path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def _save_mesh_ply_manually(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray]
):
    """Manually save mesh PLY file."""
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        for i, v in enumerate(vertices):
            line = f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
            if colors is not None:
                c = colors[i]
                if c.max() <= 1.0:
                    c = (c * 255).astype(int)
                line += f" {int(c[0])} {int(c[1])} {int(c[2])}"
            f.write(line + "\n")
            
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def load_image(path: str, mode: str = 'RGB') -> np.ndarray:
    """
    Load image from file.
    
    Args:
        path: Image path
        mode: Color mode ('RGB', 'GRAY')
        
    Returns:
        Image array [H, W, C] or [H, W]
    """
    try:
        from PIL import Image
        
        img = Image.open(path)
        if mode == 'GRAY':
            img = img.convert('L')
        elif mode == 'RGB':
            img = img.convert('RGB')
            
        return np.array(img)
        
    except ImportError:
        import cv2
        
        img = cv2.imread(path)
        if mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        return img


def save_image(path: str, image: np.ndarray):
    """
    Save image to file.
    
    Args:
        path: Output path
        image: Image array [H, W, C] or [H, W]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to uint8 if needed
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).clip(0, 255).astype(np.uint8)
        
    try:
        from PIL import Image
        
        img = Image.fromarray(image)
        img.save(path)
        
    except ImportError:
        import cv2
        
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        cv2.imwrite(str(path), image)
