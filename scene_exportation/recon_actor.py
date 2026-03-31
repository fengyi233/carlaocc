"""
Use right-handed coordinate system. Use METER in OBJ files.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, List, TypedDict, Any

import hydra
import numpy as np
import open3d as o3d
import trimesh
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

from utils.save_utils import formatter
from utils.mesh_ops import export_mesh, merge_meshes, split_meshes_by_vertices, voxelize_mesh_simple
from utils.transforms import (
    transform_mat_obj_to_world,
    transform_mat_obj_to_carla,
)
from utils.labels import name2label
from utils.math_utils import bbox_intersects, bbox_contains_point

# Actor information structure definition
ActorInfo = TypedDict('ActorInfo', {
    'idx': int,  # Actor index in the scene
    'instance_id': int,  # Instance ID (0 for stuff classes, >0 for instance classes)
    'name': str,  # Actor name
    'label': str,  # Actor label in UE editor
    'exported_file': str,  # Exported OBJ file name
    'semantic_label': str,  # Semantic label (e.g., 'Road', 'Building', 'Vehicle')
    'ue_class': str,  # Actor class (e.g., 'StaticMeshActor', 'InstancedFoliageActor')
    'vertex_count': int,  # Number of vertices in the mesh
    'face_count': int,  # Number of faces in the mesh
    'center': List[float],  # Bounding box center [x, y, z]
    'bbox_min': List[float],  # Bounding box minimum corner [x, y, z]
    'bbox_max': List[float],  # Bounding box maximum corner [x, y, z]
    'dimensions': List[float],  # Bounding box extents [x, y, z]
    'has_voxelization': bool,  # Whether voxelization has been computed
}, total=False)  # total=False means all fields are optional, but required fields should always be present


def fbx_to_obj(fbx_path, obj_path, scale=0.01, transform_mat=None, target_number_of_triangles=None):
    if os.path.exists(obj_path):
        return
    os.makedirs(os.path.dirname(obj_path), exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(fbx_path)
    if target_number_of_triangles is not None and len(mesh.triangles) > target_number_of_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_number_of_triangles)

    mesh = trimesh.Trimesh(mesh.vertices, mesh.triangles)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    if scale != 1.0:
        mesh.apply_scale(scale)  # cm to m
    if transform_mat is not None:
        mesh.apply_transform(transform_mat)
    export_mesh(mesh, obj_path)


def voxelize_mesh(actor_mesh: trimesh.Trimesh, voxel_size: float, output_path: Path):
    """
    Voxelize an actor mesh and save to npz format.

    Args:
        actor_mesh: Mesh in world coordinates
        voxel_size: Voxel size for this actor
        output_path: Output path for voxelization

    Returns:
        True if voxelization succeeded, False otherwise
    """
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    voxel_coords, voxel_origin = voxelize_mesh_simple(actor_mesh, voxel_size)
    np.savez_compressed(
        output_path,
        voxel_coords=voxel_coords.astype(np.uint16),
        voxel_origin=voxel_origin.astype(np.float32),
        voxel_size=np.float32(voxel_size)
    )


class SceneParser:
    """Parsing unreal engine scene from exported data."""

    def __init__(self, export_scene_dir, cfg: DictConfig):
        self.export_scene_dir = Path(export_scene_dir)
        self.town_name = self.export_scene_dir.name
        self.cfg = cfg

        with open(self.export_scene_dir / "exported_scene_info.json", "r") as f:
            self.scene_data = json.load(f)

        self.exported_actors_dir = Path(self.scene_data["exported_actors_dir"])

        export_dir = Path(cfg.export_dir)
        self.fg_actors_dir = export_dir / "fg_actors"
        self.fg_actor_occ_dir = export_dir / "fg_actor_occ"
        self.fg_actor_occ_dir.mkdir(parents=True, exist_ok=True)

        self.actors: List[Dict[str, Any]] = self.scene_data["actors"]
        self.actor_infos: List[ActorInfo] = []  # List of actor information dictionaries
        self.semantic_instance_counter: Dict[str, int] = {}  # Counter for instance IDs per semantic class

        # Configuration parameters
        self.bg_voxel_size = cfg.voxelization.bg_voxel_size
        self.fg_voxel_size = cfg.voxelization.fg_voxel_size
        self.large_actor_grid_size = cfg.large_actor_split.grid_size

        # Voxelization region of interest (town-specific)
        town_config = cfg.towns.get(self.town_name, None)
        if town_config and 'region' in town_config:
            region = town_config['region']
            # Store region as numpy arrays for efficient checking
            self.region_min = np.array(region['min'])
            self.region_max = np.array(region['max'])
        else:
            self.region_min = np.array([-1e6, -1e6, -1e6])
            self.region_max = np.array([1e6, 1e6, 1e6])

        # Output paths
        self.output_obj_path = self.export_scene_dir / f"{self.town_name}.obj"
        self.output_info_path = self.export_scene_dir / "actor_info.json"
        self.bg_actors_dir = self.export_scene_dir / "bg_actors"
        self.bg_actor_occ_dir = self.export_scene_dir / "bg_actor_occ"
        self.bg_actor_occ_dir.mkdir(parents=True, exist_ok=True)

    def load_mesh(self, mesh_path) -> Optional[trimesh.Trimesh]:
        """
        Load a mesh and coordinate system transformation.

        Args:
            mesh_path: Path to the mesh file

        Returns:
            Loaded mesh in CARLA coordinate system, or None if empty
        """
        mesh = trimesh.load_mesh(str(mesh_path))
        if mesh.is_empty:
            return None
        mesh.apply_transform(transform_mat_obj_to_world)

        return mesh

    def reconstruct_whole_scene(self):
        """Merge all actor meshes into a whole scene."""
        mesh_path_list = [self.bg_actors_dir / f"{actor['name']}.obj" for actor in self.actor_infos]
        merge_meshes(
            mesh_path_list,
            output_path=self.output_obj_path
        )

    def should_split_large_actor(self, actor_mesh: trimesh.Trimesh) -> bool:
        """
        Check if an actor should be split based on its size.
        
        Args:
            actor_mesh: Mesh in world coordinates
            
        Returns:
            True if X_range and Y_range are both > grid_size (50m)
        """
        if actor_mesh is None or actor_mesh.is_empty:
            return False

        # Get bounding box extents
        extents = actor_mesh.extents  # [x, y, z]

        # Check if both X and Y ranges exceed grid_size
        return extents[0] > self.large_actor_grid_size and extents[1] > self.large_actor_grid_size

    def split_large_actor_mesh(
            self,
            actor: Dict,
            actor_mesh: trimesh.Trimesh
    ) -> list:
        """
        Split a large actor mesh into smaller chunks using spatial grid partitioning.
        Each chunk corresponds to a grid cell in the XY plane.
        
        Args:
            actor: Actor dictionary
            actor_mesh: Mesh in world coordinates
            
        Returns:
            List of (chunk_name, chunk_mesh) tuples
        """
        if actor_mesh is None or actor_mesh.is_empty:
            return []

        # Use split_meshes_by_vertices to split by spatial grid
        chunk_meshes = split_meshes_by_vertices(actor_mesh, grid_size=self.large_actor_grid_size)

        chunks = []

        for chunk_idx, chunk_mesh in enumerate(chunk_meshes):
            if not chunk_mesh.is_empty:
                # Get grid cell information from metadata
                grid_cell = chunk_mesh.metadata.get('grid_cell', (0, 0))
                chunk_name = f"{actor['name']}_split_x{grid_cell[0]:03d}_y{grid_cell[1]:03d}"
                chunks.append((chunk_name, chunk_mesh))
        return chunks

    def process_large_actor(
            self,
            actor: Dict,
            actor_mesh: trimesh.Trimesh,
            actor_idx: int
    ):
        """
        Process a large actor by splitting it into grid-based chunks.
        This applies to both Landscape actors and any other large actors.
        
        Args:
            actor: Actor dictionary
            actor_mesh: Mesh in world coordinates
            actor_idx: Index of the actor
        """
        # Split into chunks
        chunks = self.split_large_actor_mesh(actor, actor_mesh)

        if not chunks:
            raise ValueError(f"No chunks generated for {actor['name']}")

        # Save each chunk as a separate actor
        for chunk_name, chunk_mesh in chunks:
            # Check if chunk is within voxelization region
            if not bbox_intersects(chunk_mesh.bounds[0], chunk_mesh.bounds[1], self.region_min, self.region_max):
                continue

            chunk_info: ActorInfo = {
                "idx": actor_idx,
                "instance_id": 0,  # Split chunks are treated as stuff class
                "name": chunk_name,
                "label": actor["label"],
                "exported_file": f"{chunk_name}.obj",
                "semantic_label": actor["semantic_label"],
                "ue_class": actor['ue_class'],
                "vertex_count": len(chunk_mesh.vertices),
                "face_count": len(chunk_mesh.faces),
                "center": chunk_mesh.bounding_box.centroid.tolist(),
                "bbox_min": chunk_mesh.bounds[0].tolist(),
                "bbox_max": chunk_mesh.bounds[1].tolist(),
                "dimensions": chunk_mesh.extents.tolist(),
                "has_voxelization": False,  # Will be updated later
            }

            # Apply inverse transform to save in OBJ coordinate system
            chunk_mesh.apply_transform(np.linalg.inv(transform_mat_obj_to_world))
            export_mesh(chunk_mesh, output_path=self.bg_actors_dir / f"{chunk_name}.obj")
            self.actor_infos.append(chunk_info)

    def assign_instance_id(self, semantic_label: str) -> int:
        """
        Get or assign instance ID for a semantic label.

        Args:
            semantic_label: Semantic label

        Returns:
            Instance ID (0 for stuff classes, >0 for instance classes)
        """
        if not name2label[semantic_label].is_instance:
            return 0  # Stuff class: instance_id = 0

        # Instance class: assign unique instance_id
        if semantic_label not in self.semantic_instance_counter:
            self.semantic_instance_counter[semantic_label] = 0
        self.semantic_instance_counter[semantic_label] += 1
        return self.semantic_instance_counter[semantic_label]

    def save_actor_info(self) -> None:
        """
        Save actor info with statistics to file.
        """
        instance_statistics = {
            "total_actors": len(self.actor_infos),
            "instance_counts": dict(self.semantic_instance_counter),
            "total_instances": sum(self.semantic_instance_counter.values())
        }

        output_data = {
            "statistics": instance_statistics,
            "actors": self.actor_infos
        }

        with open(self.output_info_path, "w") as f:
            json.dump(formatter(output_data), f, indent=4)

        print(f"Actor info saved to {self.output_info_path}")

    def is_actor_in_valid_region(self, actor_info: ActorInfo) -> bool:
        """
        Check if an actor's bounding box intersects with the voxelization region.
        
        Args:
            actor_info: Actor info dictionary containing 'bbox_min' and 'bbox_max' fields
            
        Returns:
            True if actor intersects with region (or region is not defined), False otherwise
        """
        bbox_min = actor_info.get('bbox_min')
        bbox_max = actor_info.get('bbox_max')

        if bbox_min is None or bbox_max is None:
            raise ValueError(f"Actor {actor_info.get('name', 'unknown')} has no bbox information")

        # Check if actor bbox intersects with region bbox
        return bbox_intersects(bbox_min, bbox_max, self.region_min, self.region_max)

    def gen_actor_mesh(self, actor: Dict) -> trimesh.Trimesh:
        """
        Generate a combined mesh for an actor from its components.

        Args:
            actor: Actor dictionary from scene configuration

        Returns:
            Combined mesh in the OBJ coordinate system
        """
        fbx_path = self.exported_actors_dir / f"{actor['name']}.fbx"
        obj_path = self.bg_actors_dir / f"{actor['name']}.obj"

        # Convert FBX to OBJ with optional mesh simplification
        if self.cfg.mesh_simplification.enabled:
            target_triangles = self.cfg.mesh_simplification.target_triangles
            fbx_to_obj(fbx_path, obj_path, target_number_of_triangles=target_triangles)
        else:
            fbx_to_obj(fbx_path, obj_path)

        actor_mesh = self.load_mesh(obj_path)
        return actor_mesh

    def reconstruct_bg_actors(self) -> None:
        """Generate meshes and metadata for all actors in the scene."""
        print("\n" + "=" * 80)
        print("Processing background actors...")
        print("=" * 80)
        p_bar = tqdm(self.actors, desc="Processing background actors")
        for idx, actor in enumerate(p_bar):
            p_bar.set_description(desc=f"Generating info and mesh for actor: [{actor['name']}]")

            if actor['ue_class'] == 'InstancedFoliageActor':
                self.reconstruct_InstancedFoliageActor(actor, idx)
                continue
            # Generate actor mesh
            actor_mesh = self.gen_actor_mesh(actor)
            if actor_mesh is None or actor_mesh.is_empty:
                continue

            if not bbox_intersects(actor_mesh.bounds[0], actor_mesh.bounds[1], self.region_min, self.region_max):
                continue

            if self.should_split_large_actor(actor_mesh):
                self.process_large_actor(actor, actor_mesh, idx)
                continue

            actor_info: ActorInfo = {
                "idx": idx,
                "instance_id": self.assign_instance_id(actor["semantic_label"]),
                "name": actor["name"],
                "label": actor["label"],
                "exported_file": f"{actor['name']}.obj",
                "semantic_label": actor["semantic_label"],
                "ue_class": actor['ue_class'],
                "vertex_count": len(actor_mesh.vertices),
                "face_count": len(actor_mesh.faces),
                "center": actor_mesh.bounding_box.centroid.tolist(),
                "bbox_min": actor_mesh.bounds[0].tolist(),
                "bbox_max": actor_mesh.bounds[1].tolist(),
                "dimensions": actor_mesh.extents.tolist(),
                "has_voxelization": False,  # Will be updated later
            }
            self.actor_infos.append(actor_info)

        # Voxelize all actors
        if self.cfg.voxelization.enabled:
            self.voxelize_all_actors()

        # Save actor info with statistics
        self.save_actor_info()

    def voxelize_all_actors(self) -> None:
        """
        Voxelize all actors after OBJ files have been generated.
        Updates the actor_infos with voxelization status.
        Only voxelizes actors within the specified region.
        """
        print("\n" + "=" * 80)
        print("Voxelizing all actors...")
        print("=" * 80)
        p_bar = tqdm(self.actor_infos)
        for actor_info in p_bar:
            actor_name = actor_info['name']
            output_path = self.bg_actor_occ_dir / f"{actor_name}.npz"
            if output_path.exists():
                actor_info['has_voxelization'] = True
                continue

            # Check if actor is within voxelization region
            if not self.is_actor_in_valid_region(actor_info):
                actor_info['has_voxelization'] = False
                continue

            p_bar.set_description(desc=f"Voxelizing actor: [{actor_name}]")
            # Load the mesh from OBJ file
            obj_path = self.bg_actors_dir / actor_info['exported_file']
            actor_mesh = self.load_mesh(obj_path)

            if actor_mesh is None or actor_mesh.is_empty:
                actor_info['has_voxelization'] = False
            else:
                # Perform voxelization with background voxel size
                voxelize_mesh(actor_mesh, self.bg_voxel_size, output_path)
                actor_info['has_voxelization'] = True

    def _extract_world_pos_from_transform(self, transform_data: Dict) -> np.ndarray:
        """Extract world position from transform data."""
        return np.array([
            transform_data['location']['x'] * 0.01,
            -transform_data['location']['y'] * 0.01,
            transform_data['location']['z'] * 0.01
        ])

    def reconstruct_InstancedFoliageActor(self, actor: Dict, actor_idx: int):
        """
        Reconstruct InstancedFoliageActor with proper coordinate system transformation.
        Each instance is placed according to its transform (location, rotation, scale).
        
        Args:
            actor: Actor dictionary containing components and instances
            actor_idx: Index of the parent actor
        """
        for component in actor['components']:
            # Get semantic_label from component (not from actor)
            semantic_label = component['semantic_label']

            # Initialize counter for this semantic label
            if semantic_label not in self.semantic_instance_counter:
                self.semantic_instance_counter[semantic_label] = 0

            asset_name = Path(component['asset_path']).stem
            export_path = self.exported_actors_dir / f"{actor['name']}_{asset_name}.fbx"
            instances = component['instances']
            instance_name_prefix = export_path.stem

            # Aggregate 'grass' into a single mesh
            if any(name in instance_name_prefix.lower() for name in ['grass']):
                # Load the base mesh (keep in local/OBJ coordinate system)
                base_mesh = o3d.io.read_triangle_mesh(export_path)
                base_mesh = trimesh.Trimesh(base_mesh.vertices, base_mesh.triangles)
                base_mesh.merge_vertices()
                base_mesh.update_faces(base_mesh.unique_faces())
                base_mesh.remove_unreferenced_vertices()
                base_mesh.apply_scale(0.01)  # cm to m

                # Aggregate all instances into a single mesh
                aggregated_meshes = []
                for inst_idx, instance in enumerate(
                        tqdm(instances, desc=f"Generating {instance_name_prefix} instances meshes")):
                    transform_data = instance['transform']

                    # Check if instance is in region
                    world_pos = self._extract_world_pos_from_transform(transform_data)
                    if not bbox_contains_point(self.region_min, self.region_max, world_pos):
                        continue

                    # Build transformation matrix (already includes coordinate conversion)
                    transform_mat = self.build_transform_matrix(**transform_data)

                    # Copy base mesh, apply instance transform, then world coordinate transform
                    instance_mesh = base_mesh.copy()
                    instance_mesh.apply_transform(transform_mat)
                    instance_mesh.apply_transform(transform_mat_obj_to_world)
                    aggregated_meshes.append(instance_mesh)

                # Combine all meshes into one
                print(f"Combining {len(aggregated_meshes)} meshes...")
                combined_mesh = trimesh.util.concatenate(aggregated_meshes)

                # Create a pseudo actor dict for process_large_actor
                pseudo_actor = {
                    'name': f"{instance_name_prefix}",
                    'label': actor['label'],
                    'semantic_label': semantic_label,
                    'ue_class': actor['ue_class']
                }

                # Process as large actor
                if self.should_split_large_actor(combined_mesh):
                    self.process_large_actor(pseudo_actor, combined_mesh, actor_idx)
                continue  # Skip individual instance processing

            # Process normal components
            for inst_idx, instance in enumerate(
                    tqdm(instances, desc=f"Generating instance meshes of {instance_name_prefix}")):
                transform = instance['transform']
                world_pos = self._extract_world_pos_from_transform(transform)
                if not bbox_contains_point(self.region_min, self.region_max, world_pos):
                    continue

                # Build transformation matrix (UE to Mesh coordinate conversion)
                transform_mat = self.build_transform_matrix(**transform)

                # Generate unique obj file for each instance
                instance_name = f"{instance_name_prefix}_{inst_idx:04d}"
                instance_path = self.bg_actors_dir / f"{instance_name}.obj"
                if not instance_path.exists():
                    fbx_to_obj(export_path, instance_path, transform_mat=transform_mat)

                # Load the instance mesh and generate actor info
                instance_mesh = self.load_mesh(instance_path)

                if instance_mesh is not None:
                    # Generate actor info for this instance
                    instance_info: ActorInfo = {
                        "idx": actor_idx,
                        "instance_id": 0,  # vegetation is non-instance (stuff class)
                        "name": instance_name,
                        "label": actor["label"],
                        "exported_file": f"{instance_name}.obj",
                        "semantic_label": semantic_label,
                        "ue_class": actor['ue_class'],
                        "vertex_count": len(instance_mesh.vertices),
                        "face_count": len(instance_mesh.faces),
                        "center": instance_mesh.bounding_box.centroid.tolist(),
                        "bbox_min": instance_mesh.bounds[0].tolist(),
                        "bbox_max": instance_mesh.bounds[1].tolist(),
                        "dimensions": instance_mesh.extents.tolist(),
                        "has_voxelization": False,  # Will be updated later
                    }
                    self.semantic_instance_counter[semantic_label] += 1
                    self.actor_infos.append(instance_info)

    def build_transform_matrix(self, location: Dict, rotation: Dict, scale: Dict) -> np.ndarray:
        """
        Build a transformation matrix from location, rotation, and scale.
        Convert from UE left-handed coordinate system to Mesh right-handed coordinate system.
        
        Coordinate Systems:
        - UE (Left-handed): X-Forward, Y-Right, Z-Up
        - Mesh (Right-handed): X-Forward, Y-Upward, Z-Right
        - World (Right-handed): X-Forward, Y-Left, Z-Upward (applied later in load_mesh)
        
        The mesh will be scaled (cm to m) BEFORE this transform is applied.
        So this transform works in meter units.
        """
        # Step 1: Build transform in UE coordinate system (left-handed, meter units)
        # UE position in meters
        ue_pos = np.array([
            location['x'] * 0.01,  # X (Forward) in meters
            location['y'] * 0.01,  # Y (Right) in meters
            location['z'] * 0.01  # Z (Up) in meters
        ])

        # UE rotation: Roll -> Pitch -> Yaw (intrinsic XYZ in UE frame)
        # Build rotation matrix in UE left-handed system
        ue_rot = R.from_euler('xyz',
                              [rotation['roll'], rotation['pitch'], rotation['yaw']],
                              degrees=True).as_matrix()

        # UE scale
        ue_scale = np.diag([scale['x'], scale['y'], scale['z']])

        # Build UE transform matrix (left-handed)
        ue_transform = np.eye(4)
        ue_transform[:3, :3] = ue_rot @ ue_scale
        ue_transform[:3, 3] = ue_pos

        # Step 2: Convert from UE to Mesh
        mesh_transform = transform_mat_obj_to_carla @ ue_transform @ transform_mat_obj_to_carla.T

        return mesh_transform

    def reconstruct_fg_actors(self):
        """
        Generate foreground actor meshes and occupancy grids.
        """
        print("\n" + "=" * 80)
        print("Processing foreground actors...")
        print("=" * 80)
        self.recon_fg_vehicles()
        self.recon_fg_peds()

    def recon_fg_vehicles(self):
        """Convert vehicle FBX to OBJ and optionally voxelize."""
        asset_paths = sorted(self.fg_actors_dir.rglob(f"vehicle_*.fbx"))

        for asset_path in tqdm(asset_paths, desc="Processing vehicles"):
            obj_path = asset_path.parent / f"{asset_path.stem}.obj"

            # Convert FBX to OBJ with optional mesh simplification
            if self.cfg.mesh_simplification.enabled:
                target_triangles = self.cfg.mesh_simplification.target_triangles
                fbx_to_obj(asset_path, obj_path, target_number_of_triangles=target_triangles)
            else:
                fbx_to_obj(asset_path, obj_path)

            # Voxelization
            if self.cfg.voxelization.enabled:
                output_path = self.fg_actor_occ_dir / asset_path.parent.name / f"{asset_path.stem}.npz"
                if output_path.exists():
                    continue

                mesh = self.load_mesh(obj_path)
                voxelize_mesh(mesh, self.fg_voxel_size, output_path)

    def recon_fg_peds(self):
        """Convert pedestrian FBX animations to OBJ and optionally voxelize."""
        fg_ped_dir = self.fg_actors_dir / "Pedestrians"
        fg_ped_exported_dir = fg_ped_dir / 'standard_walking'

        if fg_ped_exported_dir.exists():
            print(f'Foreground pedestrian export directory already exists. Skipping pedestrian reconstruction.')
            return
        fg_ped_exported_dir.mkdir(parents=True, exist_ok=True)

        # Animation sequences
        frame_counts = json.load(open(fg_ped_dir / "frame_info.json"))
        # load AS_walking04_G3.fbx as standard human walking motion
        as_name = 'AS_walking04_G3'
        frame_count = frame_counts[as_name]
        as_path = (fg_ped_dir / f"{as_name}.fbx").as_posix()
        output_dir = (fg_ped_dir / 'standard_walking').as_posix()
        os.system((f"blender --background "
                   f"--python {os.path.dirname(__file__)}/blender.py "
                   f"{as_path} "
                   f"{output_dir} "
                   f"{frame_count} "
                   ))

        # Convert FBX to OBJ and voxelize
        asset_paths = sorted((fg_ped_dir / 'standard_walking').rglob(f"*.fbx"))

        if not asset_paths:
            raise FileNotFoundError(f"Could not find any fbx files. Check the blender export step.")

        for fbx_path in tqdm(asset_paths, desc="Processing pedestrians"):
            obj_path = fbx_path.with_suffix('.obj')
            fbx_to_obj(fbx_path, obj_path, scale=1.0)

            # Voxelize if enabled
            if self.cfg.voxelization.enabled:
                mesh = self.load_mesh(obj_path)
                if mesh is not None and not mesh.is_empty:
                    rel_path = fbx_path.relative_to(self.fg_actors_dir)
                    output_path = self.fg_actor_occ_dir / rel_path.with_suffix('.npz')
                    voxelize_mesh(mesh, self.fg_voxel_size, output_path)


@hydra.main(version_base=None, config_path="config", config_name="recon_actor")
def main(cfg: DictConfig):
    export_dir = Path(cfg.export_dir)

    for town_name in cfg.town_names:
        print(f"\nProcessing town: {town_name}")
        town_dir = export_dir / town_name

        if not town_dir.exists():
            print(f"Warning: Town directory not found: {town_dir}")
            continue

        composer = SceneParser(town_dir, cfg)

        # Process based on configuration
        if cfg.processing.reconstruct_fg_actors:
            print("Reconstructing foreground actors...")
            composer.reconstruct_fg_actors()

        if cfg.processing.reconstruct_bg_actors:
            print("Reconstructing background actors...")
            composer.reconstruct_bg_actors()

        if cfg.processing.reconstruct_whole_scene:
            print("Reconstructing whole scene...")
            composer.reconstruct_whole_scene()

        print(f"Completed processing for {town_name}")


if __name__ == "__main__":
    main()
