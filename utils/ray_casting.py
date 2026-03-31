import numpy as np
import open3d as o3d
import open3d.core as o3c


class RayCasting:
    """Ray casting class for handling ray tracing operations"""

    def __init__(self, width: int, height: int):
        """
        Initialize RayCasting class
        
        Args:
            width: Image width
            height: Image height
        """
        self.width = width
        self.height = height

    def create_triangle_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> o3d.t.geometry.TriangleMesh:
        """
        Create Open3D triangle mesh
        
        Args:
            vertices: Vertex coordinates [N, 3]
            faces: Face indices [M, 3]
            
        Returns:
            Open3D triangle mesh object
        """
        vertices_tensor = o3c.Tensor(vertices.astype(np.float32), dtype=o3c.Dtype.Float32)
        faces_tensor = o3c.Tensor(faces.astype(np.int32), dtype=o3c.Dtype.Int32)

        tmesh = o3d.t.geometry.TriangleMesh()
        tmesh.vertex.positions = vertices_tensor
        tmesh.triangle.indices = faces_tensor

        return tmesh

    def cast_rays(self, scene: o3d.t.geometry.RaycastingScene,
                  rays: o3c.Tensor) -> np.ndarray:
        """
        Execute ray casting
        
        Args:
            scene: Open3D RaycastingScene object
            rays: Precomputed ray tensor
            
        Returns:
            Depth map [H, W]
        """
        # Execute ray casting
        ans = scene.cast_rays(rays)
        depth = ans['t_hit'].numpy().astype(np.float32)

        return depth

    def cast_rays_multiple_meshes(self, scene: o3d.t.geometry.RaycastingScene,
                                  rays: o3c.Tensor) -> tuple:
        """
        Execute ray casting for multiple meshes (for semantic generation)
        
        Args:
            scene: Open3D RaycastingScene object (already contains all meshes)
            rays: Precomputed ray tensor

        Returns:
            tuple: (geometry_ids, raytraced_depth)
                - geometry_ids: Geometry ID map [H, W]
                - raytraced_depth: Ray traced depth map [H, W]
        """
        # Execute ray casting
        ans = scene.cast_rays(rays)
        geometry_ids = ans['geometry_ids'].numpy()  # [H, W]
        raytraced_depth = ans['t_hit'].numpy().astype(np.float32)  # [H, W]

        return geometry_ids, raytraced_depth
