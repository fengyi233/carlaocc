import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List

import numpy as np
import open3d as o3d
import trimesh
from tqdm import tqdm
from trimesh import Trimesh

from utils.labels import id2label


def sample_pts_from_mesh(mesh: trimesh.Trimesh, num_points: int, method: str = 'uniform') -> np.ndarray:
    if method == 'uniform':
        points, _ = trimesh.sample.sample_surface_even(mesh, count=num_points)
    elif method == 'random':
        result = trimesh.sample.sample_surface(mesh, count=num_points)
        points = result[0]
    else:
        raise NotImplementedError
    return points


def downsample_mesh(
        obj_path: str,
        sampling_method: str = "poisson",
        num_points: Optional[int] = None,
        voxel_size: Optional[float] = None,
        poisson_samples: Optional[int] = None,
        output_format: str = "pcd"
) -> Tuple[np.ndarray, str]:
    """
    Sample point cloud from a mesh file with various sampling methods.

    Args:
        obj_path: Path to the input OBJ file
        sampling_method: Sampling method ("random", "voxel", or "poisson")
        num_points: Number of points for random sampling
        voxel_size: Voxel size for voxel-based downsampling
        poisson_samples: Number of samples for Poisson disk sampling
        output_format: Output format ("pcd" or "obj")

    Returns:
        Tuple containing (sampled_points, output_file_path)

    Raises:
        ValueError: If invalid parameters are provided
        IOError: If file operations fail
    """
    # Validate input parameters
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"Input file {obj_path} does not exist")

    if sampling_method not in ["random", "voxel", "poisson"]:
        raise ValueError("Invalid sampling method. Choose from: random, voxel, poisson")

    if sampling_method == "random" and num_points is None:
        raise ValueError("num_points must be specified for random sampling")

    if sampling_method == "voxel" and voxel_size is None:
        raise ValueError("voxel_size must be specified for voxel sampling")

    if sampling_method == "poisson" and poisson_samples is None:
        raise ValueError("poisson_samples must be specified for Poisson sampling")

    if output_format not in ["pcd", "obj"]:
        raise ValueError("Output format must be either 'pcd' or 'obj'")

    # Load mesh using trimesh
    try:
        mesh = trimesh.load(obj_path)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Loaded file is not a triangle mesh")
    except Exception as e:
        raise IOError(f"Failed to load mesh: {str(e)}")

    # Convert to open3d mesh for sampling
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    # Perform sampling based on selected method
    if sampling_method == "random":
        pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=num_points)
    elif sampling_method == "voxel":
        pcd = o3d_mesh.sample_points_uniformly(number_of_points=100000)  # Initial dense sampling
        pcd = pcd.voxel_down_sample(voxel_size)
    elif sampling_method == "poisson":
        pcd = o3d.geometry.TriangleMesh.sample_points_poisson_disk(
            o3d_mesh,
            number_of_points=poisson_samples
        )

    # Get sampled points as numpy array
    sampled_points = np.asarray(pcd.points)

    # Generate output path
    input_dir = os.path.dirname(obj_path)
    input_name = os.path.splitext(os.path.basename(obj_path))[0]
    method_tag = {
        "random": f"rand_{num_points}",
        "voxel": f"voxel_{voxel_size:.4f}",
        "poisson": f"poisson_{poisson_samples}"
    }[sampling_method]

    output_filename = f"{input_name}_{method_tag}_sampled.{output_format}"
    output_path = os.path.join(input_dir, output_filename)

    # Save point cloud
    try:
        if output_format == "pcd":
            o3d.io.write_point_cloud(output_path, pcd)
        else:
            # For OBJ output, we'll just write vertices
            with open(output_path, 'w') as f:
                for point in sampled_points:
                    f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    except Exception as e:
        raise IOError(f"Failed to save output: {str(e)}")

    return sampled_points, output_path


def crop_mesh(
        mesh: trimesh.Trimesh,
        bbox: np.ndarray,
        crop_method: str = 'aabb'
) -> trimesh.Trimesh:
    bbox_min, bbox_max = bbox[0], bbox[1]

    bbox_size = bbox_max - bbox_min
    bbox_transform = np.eye(4)
    bbox_transform[:3, 3] = (bbox_min + bbox_max) / 2

    if crop_method == 'boolean':
        box = trimesh.creation.box(extents=bbox_size, transform=bbox_transform)
        mesh_cropped = trimesh.boolean.intersection([mesh, box], engine='blender', check_volume=False,
                                                    use_exact=True)
    elif crop_method == 'aabb':
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        aabb = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
        o3d_mesh = o3d_mesh.crop(aabb)
        mesh_cropped = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh.vertices),
            faces=np.asarray(o3d_mesh.triangles),
            process=False
        )
    else:
        raise NotImplementedError

    return mesh_cropped


def crop_pano_mesh(
        pano_mesh: Dict[int, Union[trimesh.Trimesh, List[tuple]]],
        bbox: np.ndarray,
        crop_method: str = 'aabb'):
    cropped_pano_mesh = deepcopy(pano_mesh)
    for sem_id, mesh_data in cropped_pano_mesh.items():
        if id2label[sem_id].is_instance:
            continue
        if isinstance(mesh_data, list):
            # Instance-level class: process each (instance_id, mesh) tuple
            cropped_instances = []
            for instance_id, mesh in mesh_data:
                if mesh is None:
                    continue
                mesh_cropped = crop_mesh(mesh, bbox, crop_method)
                if mesh_cropped.is_empty:
                    continue
                cropped_instances.append((instance_id, mesh_cropped))
            cropped_pano_mesh[sem_id] = cropped_instances
        else:
            # Stuff class: single mesh
            mesh = mesh_data
            if mesh is None:
                continue
            mesh_cropped = crop_mesh(mesh, bbox, crop_method)
            cropped_pano_mesh[sem_id] = mesh_cropped
    return cropped_pano_mesh


def crop_sem_mesh(
        sem_mesh: Dict[int, trimesh.Trimesh],
        bbox: np.ndarray,
        crop_method: str = 'aabb'):
    cropped_sem_mesh = deepcopy(sem_mesh)
    for sem_id, mesh_data in cropped_sem_mesh.items():
        if mesh_data is None:
            continue
        mesh_cropped = crop_mesh(mesh_data, bbox, crop_method)
        cropped_sem_mesh[sem_id] = mesh_cropped
    return cropped_sem_mesh


def voxelize_mesh(
        mesh: trimesh.Trimesh,
        voxel_size: float,
        bbox: np.ndarray,
        margin: float = 10,
        crop: bool = True,
        crop_method: str = 'boolean'
) -> Optional[np.ndarray]:
    bbox_min, bbox_max = bbox[0], bbox[1]
    bbox_size = bbox_max - bbox_min
    bbox_res = (bbox_size / voxel_size).astype(np.uint32)

    # Add a margin
    bbox_min, bbox_max = bbox_min - margin * voxel_size, bbox_max + margin * voxel_size
    bbox_size = bbox_max - bbox_min
    bbox_transform = np.eye(4)
    bbox_transform[:3, 3] = (bbox_min + bbox_max) / 2

    if crop:
        if crop_method == 'aabb':
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            aabb = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
            o3d_mesh = o3d_mesh.crop(aabb)
        elif crop_method == 'boolean':
            box = trimesh.creation.box(extents=bbox_size, transform=bbox_transform)
            mesh_cropped = trimesh.boolean.intersection([mesh, box], engine='blender', check_volume=False,
                                                        use_exact=True)
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_cropped.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_cropped.faces)
        else:
            raise NotImplementedError
    else:
        mesh_cropped = mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_cropped.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_cropped.faces)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(o3d_mesh, voxel_size)

    if voxel_grid.has_voxels():
        voxels = voxel_grid.get_voxels()
        grid_indices = np.asarray([voxel.grid_index for voxel in voxels])
        vox0_center = voxel_grid.origin + voxel_size / 2
        voxel_centers = grid_indices * voxel_size + vox0_center
        voxel_indices = np.round((voxel_centers - bbox_min) / voxel_size).astype(np.int32)

        # Remove the margin
        voxel_indices -= margin

        in_bbox_mask = np.all(
            (voxel_indices >= 0) &
            (voxel_indices < bbox_res), axis=1
        )
        voxel_indices = voxel_indices[in_bbox_mask]

        return voxel_indices
    else:
        return None


def voxelize_mesh_simple(
        mesh: trimesh.Trimesh,
        voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(o3d_mesh, voxel_size)

    if voxel_grid.has_voxels():
        voxels = voxel_grid.get_voxels()
        grid_indices = np.asarray([voxel.grid_index for voxel in voxels], dtype=np.uint32)
        return grid_indices, voxel_grid.origin
    else:
        raise ValueError("Voxelization failed")


def clean_mesh(mesh: Trimesh) -> Trimesh:
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def export_mesh(mesh: Trimesh, output_path: Union[str, os.PathLike, Path]) -> None:
    mesh.export(
        output_path,
        file_type='obj',
        include_normals=False,
        include_color=False,
        include_texture=False,
        digits=6
    )


def merge_meshes(
        mesh_path_list: list,
        output_path: Union[str, os.PathLike, Path],
        mesh_vertex_num_list: Optional[list] = None
) -> None:
    """
    Merge multiple OBJ files into a single mesh.

    Args:
        mesh_path_list: List of input OBJ paths
        output_path: Output file path
        mesh_vertex_num_list: Optional list of vertex counts

    Note:
        It may be inaccurate to use `len(Trimesh.vertices)` to get vertex counts.
        It is recommended to directly compute the vertex counts from the OBJ file.
    """
    # Count vertices if not provided
    if mesh_vertex_num_list is None:
        mesh_vertex_num_list = []
        for mesh_path in tqdm(mesh_path_list, desc="Counting vertices"):
            with open(mesh_path, 'r') as f:
                mesh_vertex_num_list.append(
                    sum(1 for line in f if line.startswith('v '))
                )

    # Calculate vertex offsets for face indices
    vertex_offsets = np.cumsum([0] + mesh_vertex_num_list[:-1])

    # Merge meshes
    with open(output_path, 'w') as out_file:
        for i, mesh_path in enumerate(tqdm(mesh_path_list, desc="Merging meshes")):
            out_file.write(f"# mesh_{i}_{os.path.basename(mesh_path)}\n")
            offset = vertex_offsets[i]

            with open(mesh_path, 'r') as in_file:
                for line in in_file:
                    if line.startswith('v '):
                        out_file.write(line)
                    elif line.startswith('f '):
                        # Adjust face indices with offset
                        parts = line.strip().split()
                        adjusted_face = ['f']
                        for part in parts[1:]:
                            indices = part.split('/')
                            # Adjust vertex index
                            adjusted_idx = str(int(indices[0]) + offset)
                            if len(indices) > 1:
                                adjusted_idx += '/' + '/'.join(indices[1:])
                            adjusted_face.append(adjusted_idx)
                        out_file.write(' '.join(adjusted_face) + '\n')
                    elif line.startswith(('o ', 'g ')):
                        out_file.write(line)

    print(f"Successfully merged {len(mesh_path_list)} meshes into {output_path}")


def split_meshes(
        mesh: trimesh.Trimesh,
        faces_per_chunk: int = 10000,
) -> list:
    """
    Split a large mesh into smaller chunks by dividing faces.
    Each chunk contains at most faces_per_chunk faces.

    Args:
        mesh: Mesh to split (in world coordinates)
        faces_per_chunk: Maximum number of faces per chunk

    Returns:
        List of chunk meshes
    """
    if mesh is None or mesh.is_empty:
        return []

    total_faces = len(mesh.faces)
    num_chunks = int(np.ceil(total_faces / faces_per_chunk))

    chunks = []

    for chunk_idx in range(num_chunks):
        # Calculate face range for this chunk
        start_face = chunk_idx * faces_per_chunk
        end_face = min(start_face + faces_per_chunk, total_faces)

        # Extract faces for this chunk
        chunk_faces = mesh.faces[start_face:end_face]

        # Get unique vertices used by these faces
        unique_vertices = np.unique(chunk_faces.flatten())

        # Create vertex mapping (old index -> new index)
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}

        # Remap faces to new vertex indices
        new_faces = np.array([[vertex_map[v] for v in face] for face in chunk_faces])
        new_vertices = mesh.vertices[unique_vertices]

        # Create chunk mesh
        chunk_mesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            process=False
        )

        if not chunk_mesh.is_empty:
            chunks.append(chunk_mesh)

    print(f"Successfully split {total_faces} faces into {len(chunks)} chunks")
    return chunks


def split_meshes_by_vertices(
        mesh: trimesh.Trimesh,
        grid_size: float = 100.0,
) -> list:
    """
    Split a large mesh into smaller chunks using 2D spatial grid partitioning (XY plane).
    Each chunk corresponds to a square grid cell in the XY plane (horizontal plane).
    
    Coordinate system: X-forward, Y-left, Z-upward
    Grid is divided on the horizontal XY plane, ignoring Z (height).
    
    Important: Each face is assigned to exactly ONE chunk based on its centroid position.
    This prevents creating duplicate faces and ensures clean mesh splitting.

    Args:
        mesh: Mesh to split (in world coordinates)
        grid_size: Size of each square grid cell in XY plane (e.g., 100 means 100x100 squares)

    Returns:
        List of chunk meshes with metadata
    """
    if mesh is None or mesh.is_empty:
        return []

    vertices = mesh.vertices
    faces = mesh.faces

    # Calculate bounding box and grid dimensions (only for XY plane)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min

    # Grid dimensions for X and Y axes (ignore Z which is height)
    # Extract X (axis 0) and Y (axis 1)
    grid_dims_xy = np.ceil(bbox_size[[0, 1]] / grid_size).astype(int)

    # Use trimesh's triangles_center property for vectorized face centroid calculation
    # This is much faster than looping through faces
    face_centroids = mesh.triangles_center  # Shape: (num_faces, 3)
    face_centroids_xy = face_centroids[:, [0, 1]]  # Extract only XY coordinates

    # Vectorized grid cell assignment for all faces
    cell_indices_xy = np.floor((face_centroids_xy - bbox_min[[0, 1]]) / grid_size).astype(int)
    cell_indices_xy = np.clip(cell_indices_xy, 0, grid_dims_xy - 1)

    # Group faces by grid cells using numpy operations
    # Convert 2D cell indices to 1D for efficient grouping
    cell_ids_1d = cell_indices_xy[:, 0] * grid_dims_xy[1] + cell_indices_xy[:, 1]

    # Get unique cells and build face groups
    unique_cell_ids = np.unique(cell_ids_1d)

    # Build cell_faces dictionary efficiently
    cell_faces = {}
    for cell_1d_id in unique_cell_ids:
        # Get all face indices belonging to this cell
        face_mask = (cell_ids_1d == cell_1d_id)
        face_indices = np.where(face_mask)[0]

        # Convert 1D cell id back to 2D
        ix = cell_1d_id // grid_dims_xy[1]
        iy = cell_1d_id % grid_dims_xy[1]
        cell_2d = (int(ix), int(iy))

        cell_faces[cell_2d] = face_indices

    # Create chunk meshes for each non-empty cell
    chunks = []

    for cell_idx, face_indices in sorted(cell_faces.items()):
        if len(face_indices) == 0:
            continue

        # Get faces for this cell
        chunk_faces = faces[face_indices]

        # Get all unique vertices used by these faces
        unique_vertex_indices = np.unique(chunk_faces.flatten())

        # Check if we have valid geometry
        if len(unique_vertex_indices) < 3:
            continue

        # Create vertex mapping (old index -> new index)
        vertex_map = np.full(len(vertices), -1, dtype=np.int32)
        vertex_map[unique_vertex_indices] = np.arange(len(unique_vertex_indices))

        # Remap faces to new vertex indices using vectorized operation
        new_faces = vertex_map[chunk_faces]
        new_vertices = vertices[unique_vertex_indices]

        # Validate that all face indices are valid
        if np.any(new_faces < 0):
            print(f"Warning: Cell {cell_idx} has invalid face indices, skipping")
            continue

        # Create chunk mesh with validation
        chunk_mesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            process=False
        )

        # Store 2D grid cell information in metadata
        chunk_mesh.metadata['grid_cell'] = cell_idx  # (ix, iy)
        chunk_mesh.metadata['grid_size'] = grid_size
        chunks.append(chunk_mesh)

    return chunks
