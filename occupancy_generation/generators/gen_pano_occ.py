"""
Panoptic occupancy generator using preprocessed background and foreground voxelizations.
This script loads precomputed voxelizations from bg_actor_occ/ and fg_actor_occ/ for faster processing.

Notes:
- Background voxelizations are stored in world coordinates
- Foreground voxelizations are stored in local coordinates
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig
from tqdm import tqdm

from occupancy_generation.generators.mesh_generator import BaseMeshGenerator, _build_animated_ped_mesh
from occupancy_generation.generators.ped_analyzer import PedGaitMatcher
from utils.carla_utils import get_vehicle_semantic_class
from utils.data_converter import encode_combined_id, get_fg_instance_id
from utils.labels import name2label, sem_z_order
from utils.mesh_ops import voxelize_mesh
from utils.occ_ops import occ_dense2sparse_fast


class PreprocessedPanoOccGenerator(BaseMeshGenerator):
    """
    Panoptic occupancy generator using preprocessed background and foreground voxelizations.

    Key optimization: All actors (background and foreground) with precomputed voxelizations
    are loaded directly from bg_actor_occ/ and fg_actor_occ/ instead of being voxelized on-the-fly.

    - Background voxelizations are stored in world coordinates
    - Foreground voxelizations are stored in local coordinates (transformed with actor pose)
    """

    def __init__(self, cfg: DictConfig, town_name: str, sequence: str):
        super().__init__(cfg, town_name, sequence)

        # Derive additional paths from inherited scene_mesh_dir
        self.bg_actors_dir = self.scene_mesh_dir / town_name / "bg_actors"
        self.bg_actor_occ_dir = self.scene_mesh_dir / town_name / "bg_actor_occ"
        self.fg_actors_dir = self.scene_mesh_dir / "fg_actors"
        self.fg_actor_occ_dir = self.scene_mesh_dir / "fg_actor_occ"
        actor_info_path = self.scene_mesh_dir / town_name / "actor_info.json"
        lidar_pose_path = self.data_dir / "poses" / "lidar.txt"
        calib_path = Path(cfg.dataset_dir) / "calib" / "calib.yaml"

        # Occupancy bounding box in the lidar coordinate system
        self.voxel_size = float(cfg.voxel_size)
        self.volume_size = ((self.occ_bbox[1] - self.occ_bbox[0]) / self.voxel_size).astype(np.int32)

        # Precompute homogeneous bbox corners for efficient transformation
        bbox_corners = np.array([
            [self.occ_bbox[0, 0], self.occ_bbox[0, 1], self.occ_bbox[0, 2]],
            [self.occ_bbox[0, 0], self.occ_bbox[0, 1], self.occ_bbox[1, 2]],
            [self.occ_bbox[0, 0], self.occ_bbox[1, 1], self.occ_bbox[0, 2]],
            [self.occ_bbox[0, 0], self.occ_bbox[1, 1], self.occ_bbox[1, 2]],
            [self.occ_bbox[1, 0], self.occ_bbox[0, 1], self.occ_bbox[0, 2]],
            [self.occ_bbox[1, 0], self.occ_bbox[0, 1], self.occ_bbox[1, 2]],
            [self.occ_bbox[1, 0], self.occ_bbox[1, 1], self.occ_bbox[0, 2]],
            [self.occ_bbox[1, 0], self.occ_bbox[1, 1], self.occ_bbox[1, 2]]
        ])
        self.homo_bbox_corners = np.column_stack([bbox_corners, np.ones(8)])  # Shape: (8, 4)

        # Load camera and LiDAR data
        self.lidar_poses = {arr[0]: arr[1:].reshape(4, 4) for arr in np.loadtxt(lidar_pose_path, dtype=np.float32)}
        with open(calib_path, 'r') as f:
            self.calib = yaml.safe_load(f)
        self.cam_to_lidar = {
            f"0{i}": np.array(self.calib['sensors'][f'cam_0{i}']['transform'])
            for i in range(6)
        }

        # Load background actor info
        with open(actor_info_path) as f:
            actor_data = json.load(f)
            self.bg_actors = actor_data['actors']
            self.statistics = actor_data.get('statistics', {})
        # Filter crosswalk actors due to its non-label semantic class
        self.bg_actors = [actor for actor in self.bg_actors if "crosswalk" not in actor["name"].lower()]
        self.bg_actors_arr = np.array(self.bg_actors, dtype=object)  # For numpy boolean indexing

        # Build bboxes for background actors
        self.bg_bboxes = np.array([[c['bbox_min'], c['bbox_max']] for c in self.bg_actors])  # [N,2,3]

        # Initialize pedestrian gait matcher (inject derived ped_dir)
        gait_cfg = dict(cfg.get('pedestrian_gait', {}))
        gait_cfg['ped_dir'] = str(self.fg_actors_dir / 'Pedestrians' / 'standard_walking')
        self.gait_analyzer = PedGaitMatcher(self.data_dir, gait_cfg)

        # Cache for loaded voxelizations and meshes
        self.voxel_cache = {}  # {actor_name: (voxel_coords, voxel_origin, voxel_size)}
        self._cached_actor_meshes = {}  # Cache for loaded actor meshes (pedestrians)
        self._world_coords_cache = {}  # {actor_name: world_coords}

    def query_bg_actors_in_view(self, lidar_pose: np.ndarray) -> List[Dict]:
        """Find background actors within occupancy bounding box.

        Args:
            lidar_pose: 4x4 transformation matrix (lidar to world)

        Returns:
            List of actor info dicts in view
        """
        # Transform precomputed bbox corners to world coordinates
        world_corners = (self.homo_bbox_corners @ lidar_pose.T)[:, :3]
        world_bbox = np.array([np.min(world_corners, axis=0), np.max(world_corners, axis=0)])

        # Query actors that intersect with this world region
        bbox_mins = self.bg_bboxes[:, 0, :]  # Shape: (N, 3)
        bbox_maxs = self.bg_bboxes[:, 1, :]  # Shape: (N, 3)

        # Intersection condition: actor.max >= world.min AND actor.min <= world.max
        intersects = (
                np.all(bbox_maxs >= world_bbox[0], axis=1) &
                np.all(bbox_mins <= world_bbox[1], axis=1)
        )
        return self.bg_actors_arr[intersects].tolist()

    def get_ped_mesh(self, actor: Dict, frame_id: int, lidar_pose: np.ndarray):
        """Get pedestrian mesh in lidar coordinates.

        Args:
            actor: Actor info dict with 'id', 'transform', 'bbox_extent'
            frame_id: Current frame ID
            lidar_pose: 4x4 lidar-to-world transformation matrix

        Returns:
            Pedestrian Trimesh in lidar coordinates
        """
        ped_id = actor.get('id', None)
        db_frame = self.gait_analyzer.analyze_pedestrian_gait(ped_id, frame_id)
        return _build_animated_ped_mesh(
            self._cached_actor_meshes, self.load_mesh,
            self.fg_actors_dir, actor, db_frame, lidar_pose
        )

    def load_precomputed_voxelization(self, actor_name: str) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Load precomputed voxelization for a background actor.

        The voxelizations are stored in world coordinates with their own bounding box.

        Args:
            actor_name: Name of the actor

        Returns:
            Tuple of (voxel_coords, voxel_origin, voxel_size) or None if not available
        """
        # Check cache first
        if actor_name in self.voxel_cache:
            return self.voxel_cache[actor_name]

        # Load from disk (from bg_actor_occ directory)
        voxel_path = self.bg_actor_occ_dir / f"{actor_name}.npz"

        if not voxel_path.exists():
            print(f"Warning: Voxelization file not found for {actor_name}: {voxel_path}")
            return None

        data = np.load(voxel_path)
        voxel_coords = data['voxel_coords']  # uint16 indices
        voxel_origin = data['voxel_origin']  # float32 world coordinates
        voxel_size = float(data['voxel_size'])  # float32 voxel size

        # Cache the loaded data
        self.voxel_cache[actor_name] = (voxel_coords, voxel_origin, voxel_size)

        return (voxel_coords, voxel_origin, voxel_size)

    def load_fg_vehicle_voxelization(
            self,
            actor_type: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Load precomputed voxelization for a foreground vehicle.

        Vehicles are organized by subfolder (Bus/Car) and have voxelizations
        in their local coordinate system.

        Args:
            actor_type: Vehicle type (e.g., 'vehicle.audi.a2')

        Returns:
            Tuple of (voxel_coords, voxel_origin, voxel_size) or None if not available
        """
        # Create cache key for vehicle
        cache_key = f"fg_{actor_type}"
        semantic_class = get_vehicle_semantic_class(actor_type)

        # Check cache first
        if cache_key in self.voxel_cache:
            return self.voxel_cache[cache_key]

        # Build file path: fg_actor_occ_dir / semantic_class / actor_name.npz
        actor_name = actor_type.replace('.', '_')
        voxel_path = self.fg_actor_occ_dir / semantic_class / f"{actor_name}.npz"

        if not voxel_path.exists():
            return None

        try:
            data = np.load(voxel_path)
            voxel_coords = data['voxel_coords']  # uint16 indices
            voxel_origin = data['voxel_origin']  # float32 local coordinates
            voxel_size = float(data['voxel_size'])  # float32 voxel size

            # Cache the loaded data
            self.voxel_cache[cache_key] = (voxel_coords, voxel_origin, voxel_size)

            return (voxel_coords, voxel_origin, voxel_size)
        except Exception as e:
            print(f"Warning: Failed to load voxelization for {actor_type}: {e}")
            return None

    def transform_voxel_coords_batch(
            self,
            batch_data: List[Tuple[np.ndarray, np.ndarray, float, str]],
            world_to_lidar: np.ndarray
    ) -> List[np.ndarray]:
        """Batch transform voxel coordinates from world space to lidar space for multiple actors.

        Args:
            batch_data: List of tuples (voxel_coords, voxel_origin, voxel_size, actor_name)
            world_to_lidar: 4x4 world-to-lidar transformation matrix

        Returns:
            List of transformed voxel coordinate arrays
        """

        # Precompute constants
        R_T = np.ascontiguousarray(world_to_lidar[:3, :3].T, dtype=np.float32)  # (3,3)
        t = np.asarray(world_to_lidar[:3, 3], dtype=np.float32)  # (3,)
        occ_min = self.occ_bbox[0]  # (3,)
        inv_vs = np.float32(1.0 / self.voxel_size)
        vol = self.volume_size

        # Step 1: Collect world coordinates with caching
        batch_world_coords = []
        actor_lengths = []  # Track length of each actor's coordinates

        for voxel_coords, voxel_origin, voxel_size, actor_name in batch_data:
            if len(voxel_coords) == 0:
                batch_world_coords.append(None)
                actor_lengths.append(0)
                continue

            # Get or compute world_coords
            if actor_name in self._world_coords_cache:
                world_coords = self._world_coords_cache[actor_name]
            else:
                world_coords = voxel_origin.astype(np.float32) + (voxel_coords.astype(np.float32) + 0.5) * voxel_size
                self._world_coords_cache[actor_name] = world_coords

            batch_world_coords.append(world_coords)
            actor_lengths.append(len(world_coords))

        # Step 2: Concatenate all world_coords for batch processing
        valid_world_coords = [wc for wc in batch_world_coords if wc is not None]

        if not valid_world_coords:
            return [np.empty((0, 3), dtype=np.int32) for _ in batch_data]

        all_world_coords = np.vstack(valid_world_coords)

        # Step 3: Batch transform world -> lidar
        all_lidar_coords = all_world_coords @ R_T
        all_lidar_coords += (t - occ_min)
        all_lidar_coords *= inv_vs
        all_lidar_voxel_coords = all_lidar_coords.astype(np.int32, copy=False)

        # Step 4: Batch bounds checking
        valid_mask = (
                (all_lidar_voxel_coords[:, 0] >= 0) & (all_lidar_voxel_coords[:, 0] < vol[0]) &
                (all_lidar_voxel_coords[:, 1] >= 0) & (all_lidar_voxel_coords[:, 1] < vol[1]) &
                (all_lidar_voxel_coords[:, 2] >= 0) & (all_lidar_voxel_coords[:, 2] < vol[2])
        )

        # Step 5: Split results back to individual actors
        results = []
        concat_start_idx = 0

        for i, length in enumerate(actor_lengths):
            if length == 0 or batch_world_coords[i] is None:
                results.append(np.empty((0, 3), dtype=np.int32))
            else:
                concat_end_idx = concat_start_idx + length
                actor_valid_mask = valid_mask[concat_start_idx:concat_end_idx]
                actor_voxel_coords = all_lidar_voxel_coords[concat_start_idx:concat_end_idx][actor_valid_mask]
                results.append(actor_voxel_coords)
                concat_start_idx = concat_end_idx

        return results

    def transform_fg_voxel_coords(
            self,
            voxel_coords: np.ndarray,
            voxel_origin: np.ndarray,
            voxel_size: float,
            actor_transform: np.ndarray,
            world_to_lidar: np.ndarray
    ) -> np.ndarray:
        """Transform foreground actor voxel coordinates from local space to lidar space.

        Foreground actors' voxelizations are stored in their local coordinate system,
        so we need to transform: local -> world -> lidar.

        Transform pipeline:
        1. Convert voxel indices to local coordinates using source voxel_size
        2. Transform from local to world coordinates using actor_transform
        3. Transform from world to lidar coordinates
        4. Convert to voxel indices in lidar space using target voxel_size

        Args:
            voxel_coords: Nx3 array of voxel indices (in actor local coordinate system)
            voxel_origin: Origin of the voxel grid in local coordinates
            voxel_size: Size of each source voxel (from precomputed data, e.g., 0.05m)
            actor_transform: 4x4 transformation matrix (local to world)
            world_to_lidar: 4x4 transformation matrix (world to lidar)

        Returns:
            Nx3 array of transformed voxel indices in lidar space (target voxel_size)
        """
        if len(voxel_coords) == 0:
            return np.empty((0, 3), dtype=np.int32)

        occ_min = self.occ_bbox[0].astype(np.float32)
        target_voxel_size = self.voxel_size
        inv_target_voxel_size = 1.0 / target_voxel_size
        volume_size = self.volume_size

        # voxel -> local
        local_coords = voxel_origin.astype(np.float32) + (voxel_coords.astype(np.float32) + 0.5) * voxel_size

        # local -> world -> lidar
        local_coords_homo = np.column_stack([local_coords, np.ones(len(local_coords), dtype=np.float32)])
        lidar_coords = world_to_lidar @ actor_transform @ local_coords_homo.T
        lidar_coords = lidar_coords.T[:, :3]

        # lidar -> voxel
        lidar_voxel_coords = ((lidar_coords - occ_min) * inv_target_voxel_size).astype(np.int32)
        valid_mask = (
                (lidar_voxel_coords[:, 0] >= 0) & (lidar_voxel_coords[:, 0] < volume_size[0]) &
                (lidar_voxel_coords[:, 1] >= 0) & (lidar_voxel_coords[:, 1] < volume_size[1]) &
                (lidar_voxel_coords[:, 2] >= 0) & (lidar_voxel_coords[:, 2] < volume_size[2])
        )
        return lidar_voxel_coords[valid_mask]

    def generate_pano_occupancy(self, frame_id: int) -> np.ndarray:
        """Generate panoptic occupancy for a single frame using preprocessed data.

        Args:
            frame_id: Frame number to process

        Returns:
            Sparse panoptic occupancy uint16 array of shape (N, 4) with columns (x, y, z, combined_id)
        """
        lidar_to_world = self.lidar_poses[frame_id]
        world_to_lidar = np.linalg.inv(lidar_to_world)

        fg_traffic_info = self.load_traffic_info(frame_id)

        # Query background actors visible in this frame
        bg_actors_info = self.query_bg_actors_in_view(lidar_to_world)

        pano_occ_dense = np.zeros(self.volume_size, dtype=np.uint16)

        # Sort semantic IDs by z_order to ensure correct layering: lower z_order rendered first
        results = []  # (sem_id, instance_id, z_order, occ_coords)

        # ========================================
        # Process background actors
        # ========================================
        batch_data = []  # Voxelization data: (voxel_coords, voxel_origin, voxel_size, actor_name)
        batch_actor_info = []  # Actor metadata: (actor_name, sem_id, instance_id, z_order)

        # Collect all actors with precomputed voxelizations for batch transformation
        for actor in bg_actors_info:
            actor_name = actor['name']
            semantic_label = actor.get('semantic_label', 'unknown')

            if semantic_label not in name2label:
                continue

            sem_id = name2label[semantic_label].id
            instance_id = actor.get('instance_id', 0)
            has_voxelization = actor.get('has_voxelization', False)

            if has_voxelization:
                voxel_data = self.load_precomputed_voxelization(actor_name)
                if voxel_data is None:
                    continue
                voxel_coords, voxel_origin, voxel_size = voxel_data
                z_order = sem_z_order.get(sem_id, 50)

                batch_data.append((voxel_coords, voxel_origin, voxel_size, actor_name))
                batch_actor_info.append((actor_name, sem_id, instance_id, z_order))
            else:
                raise ValueError(f"Actor {actor_name} has no voxelization")

        # Batch transform all background actors from world to lidar coordinates
        if batch_data:
            batch_results = self.transform_voxel_coords_batch(batch_data, world_to_lidar)

            # Collect transformed results
            for (actor_name, sem_id, instance_id, z_order), occ_coords in zip(batch_actor_info, batch_results):
                if len(occ_coords) > 0:
                    results.append((sem_id, instance_id, z_order, actor_name, occ_coords))

        # ========================================
        # Process foreground vehicles
        # ========================================
        for actor in fg_traffic_info.get('vehicles', []):
            actor_type = actor['type']
            actor_id = actor.get('id', 0)
            actor_transform = np.array(actor['transform'])

            vehicle_semantic_class = get_vehicle_semantic_class(actor_type)
            sem_id = name2label[vehicle_semantic_class].id
            instance_id = get_fg_instance_id(actor_id)

            # Load precomputed voxelization for vehicle
            voxel_data = self.load_fg_vehicle_voxelization(actor_type)

            if voxel_data is not None:
                voxel_coords, voxel_origin, voxel_size = voxel_data

                # Transform from local -> world -> lidar
                occ_coords = self.transform_fg_voxel_coords(
                    voxel_coords, voxel_origin, voxel_size,
                    actor_transform, world_to_lidar
                )

                if len(occ_coords) > 0:
                    z_order = sem_z_order.get(sem_id, 0)
                    results.append((sem_id, instance_id, z_order, actor_type, occ_coords))

        # ========================================
        # Process foreground pedestrians
        # ========================================
        # Note: Pedestrian meshes are generated on-the-fly due to size changes
        ped_id = name2label['Pedestrian'].id
        for actor in fg_traffic_info.get('pedestrians', []):
            actor_id = actor.get('id', 0)
            instance_id = get_fg_instance_id(actor_id)

            # Generate pedestrian mesh in lidar coordinates with gait animation
            ped_mesh = self.get_ped_mesh(actor, frame_id, lidar_to_world)

            if ped_mesh is not None and not ped_mesh.is_empty:
                # Voxelize mesh in lidar space
                occ_coords = voxelize_mesh(ped_mesh, self.voxel_size, bbox=self.occ_bbox, crop=False)

                if occ_coords is not None and len(occ_coords) > 0:
                    z_order = sem_z_order.get(ped_id, 50)
                    results.append((ped_id, instance_id, z_order, f"ped_{actor_id}", occ_coords))

        # ========================================
        # Merge all results into dense occupancy grid
        # ========================================
        # Sort by z_order: lower values render first, higher values overwrite
        results_sorted = sorted(results, key=lambda x: (x[2], x[3]))

        X, Y, Z = pano_occ_dense.shape

        # Define ground-like semantic classes that should clear occlusion
        CLEAR_OCCLUSIVE_SEM_IDS = {
            name2label['Sidewalk'].id,
            name2label['Road'].id,
        }

        for sem_id, instance_id, _, _, occ_coords in results_sorted:
            if occ_coords is None or occ_coords.size == 0:
                continue

            combined_id = encode_combined_id(np.uint16(sem_id), np.uint16(instance_id))
            xs = occ_coords[:, 0]
            ys = occ_coords[:, 1]
            zs = occ_coords[:, 2]

            # Special handling for ground-like surfaces (Road, Sidewalk)
            # Purpose: Remove occluded voxels above and below these surfaces to prevent
            # artifacts from incorrectly reconstructed geometry
            if sem_id in CLEAR_OCCLUSIVE_SEM_IDS:
                # Step 1: Clear 1 voxel above the surface
                zs_up = zs + 1
                valid_up = zs_up < Z  # Prevent out-of-bounds
                if np.any(valid_up):
                    pano_occ_dense[xs[valid_up], ys[valid_up], zs_up[valid_up]] = 0

                # Step 2: Clear entire column below the surface
                xy_id = xs * Y + ys
                unique_xy, inverse = np.unique(xy_id, return_inverse=True)

                # Find minimum z for each unique (x,y) position (lowest surface layer)
                z_min = np.full(unique_xy.shape[0], Z, dtype=np.int32)
                np.minimum.at(z_min, inverse, zs)

                # Recover individual x,y coordinates
                ux = unique_xy // Y
                uy = unique_xy % Y

                # Extract entire z-column for these (x,y) positions, shape: (#U, Z)
                col_grid = pano_occ_dense[ux, uy, :]  # Note: this is a copy

                # Create mask: for each (x,y), zero out positions where z < z_min
                depth = np.arange(Z)[None, :]  # Shape: (1, Z)
                mask_below = depth < z_min[:, None]  # Shape: (#U, Z)

                col_grid[mask_below] = 0
                pano_occ_dense[ux, uy, :] = col_grid  # Write back

            # Finally, write current semantic voxels
            pano_occ_dense[xs, ys, zs] = combined_id

        # Convert dense grid to sparse Nx4 format (x, y, z, combined_id)
        pano_occ_sparse = occ_dense2sparse_fast(pano_occ_dense).astype(np.uint16)
        return pano_occ_sparse


@hydra.main(config_path="../config", config_name="gen_pano_occ", version_base=None)
def main(cfg: DictConfig) -> None:
    for town_name in cfg.town_names:
        for sequence in cfg.sequences:
            print(f"\nProcessing {town_name} Seq{sequence}")
            occ_generator = PreprocessedPanoOccGenerator(cfg, town_name, sequence)

            # Set up save directory
            save_dir = occ_generator.data_dir / "occupancy" / f"vs_{str(cfg.voxel_size).replace('.', '_')}"
            save_dir.mkdir(parents=True, exist_ok=True)

            for frame_id in tqdm(range(cfg.frame_range[0], cfg.frame_range[1] + 1),
                                 desc=f"{town_name}_Seq{sequence}"):
                pano_occ_sparse = occ_generator.generate_pano_occupancy(frame_id)

                # Save compressed occupancy to disk
                save_path = save_dir / f"{frame_id:04d}.npz"
                np.savez_compressed(
                    save_path,
                    occupancy=pano_occ_sparse,
                    voxel_size=float(cfg.voxel_size),
                    voxel_origin=np.array(cfg.occ_bbox.min, dtype=np.float32),
                    volume_size=occ_generator.volume_size,
                )


if __name__ == '__main__':
    main()
