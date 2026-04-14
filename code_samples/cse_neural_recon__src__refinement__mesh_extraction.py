"""
Mesh extraction from neural implicit representations.
"""

from typing import Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class MeshData:
    """Container for mesh data."""
    vertices: np.ndarray  # [V, 3]
    faces: np.ndarray     # [F, 3]
    normals: Optional[np.ndarray] = None  # [V, 3]
    colors: Optional[np.ndarray] = None   # [V, 3] or [V, 4]
    
    @property
    def num_vertices(self) -> int:
        return len(self.vertices)
        
    @property
    def num_faces(self) -> int:
        return len(self.faces)
        
    def save(self, path: str):
        """Save mesh to file (PLY or OBJ format)."""
        if path.endswith('.ply'):
            self._save_ply(path)
        elif path.endswith('.obj'):
            self._save_obj(path)
        else:
            raise ValueError(f"Unsupported format: {path}")
            
    def _save_ply(self, path: str):
        """Save as PLY file."""
        with open(path, 'w') as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {self.num_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if self.normals is not None:
                f.write("property float nx\n")
                f.write("property float ny\n")
                f.write("property float nz\n")
            if self.colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write(f"element face {self.num_faces}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Vertices
            for i in range(self.num_vertices):
                v = self.vertices[i]
                line = f"{v[0]} {v[1]} {v[2]}"
                if self.normals is not None:
                    n = self.normals[i]
                    line += f" {n[0]} {n[1]} {n[2]}"
                if self.colors is not None:
                    c = self.colors[i]
                    if c.max() <= 1.0:
                        c = (c * 255).astype(int)
                    line += f" {int(c[0])} {int(c[1])} {int(c[2])}"
                f.write(line + "\n")
                
            # Faces
            for face in self.faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
                
    def _save_obj(self, path: str):
        """Save as OBJ file."""
        with open(path, 'w') as f:
            # Vertices
            for v in self.vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                
            # Normals
            if self.normals is not None:
                for n in self.normals:
                    f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
                    
            # Faces (1-indexed)
            for face in self.faces:
                if self.normals is not None:
                    f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
                else:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


class MarchingCubesExtractor:
    """
    Extract mesh from SDF using Marching Cubes algorithm.
    
    Args:
        resolution: Grid resolution for marching cubes
        threshold: SDF threshold for surface extraction
        bounds: Volume bounds ((min_x, min_y, min_z), (max_x, max_y, max_z))
    """
    
    def __init__(
        self,
        resolution: int = 256,
        threshold: float = 0.0,
        bounds: Optional[Tuple[Tuple[float, float, float], 
                               Tuple[float, float, float]]] = None
    ):
        self.resolution = resolution
        self.threshold = threshold
        self.bounds = bounds or ((-1, -1, -1), (1, 1, 1))
        
    def extract(
        self,
        sdf_func,
        batch_size: int = 32768
    ) -> MeshData:
        """
        Extract mesh from SDF function.
        
        Args:
            sdf_func: Function that takes [N, 3] points and returns [N] SDF values
            batch_size: Batch size for SDF evaluation
            
        Returns:
            Extracted mesh
        """
        try:
            from skimage import measure
        except ImportError:
            raise ImportError("scikit-image required for marching cubes")
            
        # Create grid
        min_bound = np.array(self.bounds[0])
        max_bound = np.array(self.bounds[1])
        
        x = np.linspace(min_bound[0], max_bound[0], self.resolution)
        y = np.linspace(min_bound[1], max_bound[1], self.resolution)
        z = np.linspace(min_bound[2], max_bound[2], self.resolution)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        
        # Evaluate SDF in batches
        sdf_values = []
        for i in range(0, len(grid_points), batch_size):
            batch_points = grid_points[i:i+batch_size]
            batch_sdf = sdf_func(batch_points)
            sdf_values.append(batch_sdf)
            
        sdf_values = np.concatenate(sdf_values)
        sdf_grid = sdf_values.reshape(self.resolution, self.resolution, self.resolution)
        
        # Run marching cubes
        try:
            vertices, faces, normals, _ = measure.marching_cubes(
                sdf_grid,
                level=self.threshold,
                spacing=(
                    (max_bound[0] - min_bound[0]) / (self.resolution - 1),
                    (max_bound[1] - min_bound[1]) / (self.resolution - 1),
                    (max_bound[2] - min_bound[2]) / (self.resolution - 1)
                )
            )
        except Exception as e:
            raise RuntimeError(f"Marching cubes failed: {e}")
            
        # Transform to world coordinates
        vertices = vertices + min_bound
        
        return MeshData(
            vertices=vertices.astype(np.float32),
            faces=faces.astype(np.int32),
            normals=normals.astype(np.float32)
        )


class SDFMeshExtractor:
    """
    Extract mesh from neural SDF with additional refinement.
    
    Features:
    - Multi-resolution extraction
    - Vertex refinement using gradient descent
    - Normal estimation from SDF gradients
    """
    
    def __init__(
        self,
        base_resolution: int = 128,
        refinement_steps: int = 3,
        refinement_lr: float = 0.1,
        bounds: Optional[Tuple[Tuple[float, float, float], 
                               Tuple[float, float, float]]] = None
    ):
        self.base_resolution = base_resolution
        self.refinement_steps = refinement_steps
        self.refinement_lr = refinement_lr
        self.bounds = bounds or ((-1, -1, -1), (1, 1, 1))
        
    def extract(
        self,
        sdf_model,
        device: str = 'cuda',
        refine_vertices: bool = True
    ) -> MeshData:
        """
        Extract and refine mesh from neural SDF.
        
        Args:
            sdf_model: Neural SDF model
            device: Device for computation
            refine_vertices: Whether to refine vertex positions
            
        Returns:
            Refined mesh
        """
        import torch
        
        # Initial extraction at base resolution
        extractor = MarchingCubesExtractor(
            resolution=self.base_resolution,
            bounds=self.bounds
        )
        
        def sdf_func(points):
            with torch.no_grad():
                pts_tensor = torch.from_numpy(points).float().to(device)
                sdf = sdf_model(pts_tensor)
                return sdf.cpu().numpy()
                
        mesh = extractor.extract(sdf_func)
        
        if not refine_vertices or self.refinement_steps == 0:
            return mesh
            
        # Refine vertices using gradient descent on SDF
        vertices = torch.from_numpy(mesh.vertices).float().to(device)
        vertices.requires_grad_(True)
        
        for _ in range(self.refinement_steps):
            sdf_values = sdf_model(vertices)
            
            # Compute gradient
            grad = torch.autograd.grad(
                sdf_values.sum(),
                vertices,
                create_graph=False
            )[0]
            
            # Project vertices onto zero level set
            with torch.no_grad():
                grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                vertices = vertices - self.refinement_lr * sdf_values.unsqueeze(-1) * grad / grad_norm
                vertices.requires_grad_(True)
                
        # Compute refined normals from SDF gradient
        with torch.enable_grad():
            vertices_final = vertices.detach().requires_grad_(True)
            sdf_final = sdf_model(vertices_final)
            normals = torch.autograd.grad(
                sdf_final.sum(),
                vertices_final
            )[0]
            normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            
        mesh.vertices = vertices.detach().cpu().numpy()
        mesh.normals = normals.detach().cpu().numpy()
        
        return mesh


def extract_mesh_from_sdf(
    sdf_model,
    resolution: int = 256,
    bounds: Optional[Tuple[Tuple[float, float, float], 
                           Tuple[float, float, float]]] = None,
    device: str = 'cuda',
    refine: bool = True
) -> MeshData:
    """
    Convenience function to extract mesh from SDF model.
    
    Args:
        sdf_model: Neural SDF model
        resolution: Grid resolution
        bounds: Volume bounds
        device: Computation device
        refine: Whether to refine vertices
        
    Returns:
        Extracted mesh
    """
    if refine:
        extractor = SDFMeshExtractor(
            base_resolution=resolution,
            bounds=bounds
        )
        return extractor.extract(sdf_model, device=device, refine_vertices=True)
    else:
        import torch
        
        extractor = MarchingCubesExtractor(
            resolution=resolution,
            bounds=bounds
        )
        
        def sdf_func(points):
            with torch.no_grad():
                pts_tensor = torch.from_numpy(points).float().to(device)
                sdf = sdf_model(pts_tensor)
                return sdf.cpu().numpy()
                
        return extractor.extract(sdf_func)
