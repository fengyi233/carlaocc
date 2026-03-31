import json
from pathlib import Path
from typing import Dict, List, OrderedDict

import numpy as np
import trimesh
import yaml
from omegaconf import DictConfig
from trimesh import Trimesh

from utils.carla_utils import get_vehicle_semantic_class
from utils.data_converter import get_fg_instance_id
from utils.labels import name2label, id2label, sem_z_order
from utils.transforms import transform_mat_obj_to_world
from .ped_analyzer import PedGaitMatcher


def _build_animated_ped_mesh(
        mesh_cache: Dict[str, Trimesh],
        load_mesh_fn,
        fg_actors_dir: Path,
        actor: Dict,
        db_frame: int,
        lidar_pose: np.ndarray
) -> Trimesh:
    """Build an animated pedestrian mesh in lidar coordinates.

    Loads the appropriate gait animation frame, applies coordinate system
    alignment, scales to match the actor's bounding box, and transforms
    from world to lidar coordinates.

    Args:
        mesh_cache: Shared cache dict for loaded meshes (modified in-place)
        load_mesh_fn: Function to load a mesh from disk (Path -> Trimesh)
        fg_actors_dir: Root directory for foreground actor mesh files
        actor: Actor info dict with 'transform' and 'bbox_extent'
        db_frame: Animation database frame index
        lidar_pose: 4x4 lidar-to-world transformation matrix

    Returns:
        Pedestrian Trimesh in lidar coordinates
    """
    obj_name = f'frame_{db_frame:02d}.obj'
    obj_path = fg_actors_dir / 'Pedestrians' / 'standard_walking' / obj_name

    if obj_name not in mesh_cache:
        mesh_cache[obj_name] = load_mesh_fn(obj_path)
    ped_mesh = mesh_cache[obj_name].copy()

    # Coordinate system alignment:
    # Animation Sequence: Z is the moving direction || Carla: X is the moving direction
    # Animation Sequence: origin at the bottom center || Carla: origin at the bbox center
    ped_mesh.apply_transform(np.array([[0, -1, 0, 0],
                                       [1, 0, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]]))

    # Shift origin from bottom center to bbox center
    bbox_min, bbox_max = ped_mesh.bounds
    mesh_extent = (bbox_max - bbox_min) / 2
    ped_mesh.apply_transform(np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, -mesh_extent[2]],
                                       [0, 0, 0, 1]]))

    # Scale to match actor's bounding box extent
    actor_extent = actor['bbox_extent']
    scale = actor_extent[2] / mesh_extent[2]
    ped_mesh.apply_scale([scale, scale, scale])

    # Transform to world coordinates, then to lidar coordinates
    ped_mesh.apply_transform(np.array(actor['transform']))
    ped_mesh.apply_transform(np.linalg.inv(lidar_pose))
    return ped_mesh


class BaseMeshGenerator:
    """Base class for mesh generators with common functionality."""

    def __init__(self, cfg: DictConfig, town_name: str, sequence: str):
        self.cfg = cfg
        dataset_dir = Path(cfg.dataset_dir)
        self.scene_mesh_dir = dataset_dir / "SceneMeshes"
        self.data_dir = dataset_dir / f"{town_name}_Seq{sequence}"
        self.town_name = town_name
        self.sequence = sequence

        # Occupancy bounding box in the lidar coordinate system
        self.occ_bbox = np.array([
            cfg.occ_bbox.min,
            cfg.occ_bbox.max
        ], dtype=np.float32)

    def load_mesh(self, obj_path: Path) -> Trimesh:
        """Load mesh and transform from OBJ coordinate system to world coordinate system.

        Args:
            obj_path: Path to the mesh file

        Returns:
            Transformed Trimesh object in world coordinates
        """
        mesh = trimesh.load_mesh(obj_path)
        mesh.apply_transform(transform_mat_obj_to_world)
        return mesh

    def load_traffic_info(self, frame_id: int) -> Dict:
        """Load traffic info for a specific frame.

        Args:
            frame_id: Frame number

        Returns:
            Traffic information dictionary
        """
        traffic_info_path = self.data_dir / "traffic_info" / f"{frame_id:04d}.yaml"
        with open(traffic_info_path) as f:
            traffic_info = yaml.safe_load(f)
        return traffic_info


class BgMeshGenerator(BaseMeshGenerator):
    """Background mesh generator with mesh caching and automatic cleanup."""

    def __init__(self, cfg: DictConfig, town_name: str, sequence: str):
        super().__init__(cfg, town_name, sequence)
        # Derive paths
        actor_info_path = self.scene_mesh_dir / town_name / "actor_info.json"
        self.bg_actors_dir = self.scene_mesh_dir / town_name / "bg_actors"

        # Load scene metadata
        with open(actor_info_path) as f:
            actor_data = json.load(f)
            self.actors = actor_data['actors']
            self.statistics = actor_data.get('statistics', {})

        # Cache configuration
        self.cache_max_size = cfg.get('bg_cache_size', 100)

        # Cache structure: {actor_name: {'mesh': Trimesh, 'last_used_frame': int, 'unused_count': int}}
        self.mesh_cache = OrderedDict()
        self.semantic_meshes = {}  # {semantic_id: {actor_name: actor_mesh}}, in world coordinate
        self.actor_instance_ids = {}  # {actor_name: instance_id}

        # Track frame numbers
        self.current_frame = 0
        self.last_cleanup_frame = 0

        self.bboxes = np.array([[c['bbox_min'], c['bbox_max']] for c in self.actors])  # [N,2,3]

    def load_mesh(self, actor: Dict) -> Trimesh:
        """Load, cache, and transform mesh to world coordinate system.

        Args:
            actor: Actor info dict containing 'name' and 'exported_file'

        Returns:
            Transformed Trimesh object in world coordinates
        """
        actor_name = actor['name']
        if actor_name in self.mesh_cache:
            cache_entry = self.mesh_cache[actor_name]

            # Update usage if used in current frame
            if cache_entry['last_used_frame'] != self.current_frame:
                cache_entry['last_used_frame'] = self.current_frame
                cache_entry['unused_count'] = 0
                self.mesh_cache.move_to_end(actor_name)  # Mark as recently used
            return cache_entry['mesh']

        # Load new mesh
        mesh = trimesh.load_mesh(self.bg_actors_dir / actor['exported_file'])
        mesh.apply_transform(transform_mat_obj_to_world)
        self.mesh_cache[actor_name] = {
            'mesh': mesh,
            'last_used_frame': self.current_frame,
            'unused_count': 0
        }
        if len(self.mesh_cache) > self.cache_max_size:
            self.mesh_cache.popitem(last=False)

        return mesh

    def gen_mesh(self, lidar_pose: np.ndarray, return_panoptic: bool = False) -> Dict:
        """Generate background meshes in lidar coordinates.

        Args:
            lidar_pose: 4x4 transformation matrix (lidar to world)
            return_panoptic: If True, return panoptic format with instance IDs

        Returns:
            If return_panoptic=True:
                {sem_id: List[(instance_id, Trimesh)]} for instance classes (is_instance=True)
                {sem_id: Trimesh} for stuff classes (is_instance=False)
            If return_panoptic=False:
                {sem_id: [mesh1, mesh2, ...]} for all classes (semantic format)
        """
        actor_ids = self._query_actors_in_view(lidar_pose, self.occ_bbox)
        current_actor_names = {self.actors[idx]['name'] for idx in actor_ids}

        # Update semantic_meshes cache
        self._update_sem_mesh_cache(current_actor_names)

        for i, idx in enumerate(actor_ids):
            actor = self.actors[idx]
            actor_name = actor['name']

            if 'crosswalk' in actor_name.lower():
                continue

            semantic_id = name2label[actor['semantic_label']].id
            if semantic_id not in self.semantic_meshes or actor_name not in self.semantic_meshes[semantic_id]:
                mesh = self.load_mesh(actor)

                if semantic_id not in self.semantic_meshes:
                    self.semantic_meshes[semantic_id] = {}
                self.semantic_meshes[semantic_id][actor_name] = mesh

                # Store instance_id for this actor
                self.actor_instance_ids[actor_name] = actor.get('instance_id', 0)

        bg_meshes = {}
        for sem_id, sem_meshes in self.semantic_meshes.items():
            if return_panoptic:
                # Check if this semantic class is instance-level
                if id2label[sem_id].is_instance:
                    # Instance class: return list of (instance_id, mesh) tuples
                    bg_meshes[sem_id] = []
                    for actor_name, mesh in sem_meshes.items():
                        instance_id = self.actor_instance_ids.get(actor_name, 0)
                        transformed_mesh = mesh.copy().apply_transform(np.linalg.inv(lidar_pose))
                        bg_meshes[sem_id].append((instance_id, transformed_mesh))
                else:
                    # Stuff class: merge into single mesh
                    transformed_meshes = [mesh.copy().apply_transform(np.linalg.inv(lidar_pose)) for mesh in
                                          sem_meshes.values()]
                    bg_meshes[sem_id] = trimesh.util.concatenate(transformed_meshes)
            else:
                # Semantic format: always return list
                transformed_meshes = [mesh.copy().apply_transform(np.linalg.inv(lidar_pose)) for mesh in
                                      sem_meshes.values()]
                bg_meshes[sem_id] = transformed_meshes

        return bg_meshes

    def _query_actors_in_view(self, lidar_pose: np.ndarray, bbox: np.ndarray = None) -> List[int]:
        """Find actors within the occupancy bounding box in world space.

        Args:
            lidar_pose: 4x4 transformation matrix (lidar to world)
            bbox: Optional bbox to use instead of self.occ_bbox (for extended view)

        Returns:
            List of component indices in view
        """
        # Use provided bbox or default to occ_bbox
        query_bbox = bbox if bbox is not None else self.occ_bbox

        # Transform bbox corners to world coordinates
        bbox_corners = np.array([
            [query_bbox[0, 0], query_bbox[0, 1], query_bbox[0, 2]],
            [query_bbox[0, 0], query_bbox[0, 1], query_bbox[1, 2]],
            [query_bbox[0, 0], query_bbox[1, 1], query_bbox[0, 2]],
            [query_bbox[0, 0], query_bbox[1, 1], query_bbox[1, 2]],
            [query_bbox[1, 0], query_bbox[0, 1], query_bbox[0, 2]],
            [query_bbox[1, 0], query_bbox[0, 1], query_bbox[1, 2]],
            [query_bbox[1, 0], query_bbox[1, 1], query_bbox[0, 2]],
            [query_bbox[1, 0], query_bbox[1, 1], query_bbox[1, 2]]
        ])

        # Transform to world coordinates
        world_corners = (lidar_pose @ np.column_stack([bbox_corners, np.ones(len(bbox_corners))]).T).T[:, :3]
        world_bbox = np.array([np.min(world_corners, axis=0), np.max(world_corners, axis=0)])

        # Query components that intersect with this world region
        component_ids = []
        for i, (bbox_min, bbox_max) in enumerate(self.bboxes):
            if np.all(bbox_max >= world_bbox[0]) and np.all(bbox_min <= world_bbox[1]):
                component_ids.append(i)

        return component_ids

    def _update_sem_mesh_cache(self, current_actor_names: set) -> None:
        """Remove meshes from semantic cache that are no longer in view.

        Args:
            current_actor_names: Set of actor names currently in view
        """
        for sem_id in list(self.semantic_meshes.keys()):
            for actor_name in list(self.semantic_meshes[sem_id].keys()):
                if actor_name not in current_actor_names:
                    del self.semantic_meshes[sem_id][actor_name]


class FgMeshGenerator(BaseMeshGenerator):
    """Foreground mesh generator for vehicles and pedestrians."""

    def __init__(self, cfg: DictConfig, town_name: str, sequence: str):
        super().__init__(cfg, town_name, sequence)
        self.fg_actors_dir = self.scene_mesh_dir / "fg_actors"
        self._cached_actor_meshes = {}  # Cache for loaded actor meshes

        # Initialize pedestrian gait matcher
        gait_cfg = dict(cfg.get('pedestrian_gait', {}))
        gait_cfg['ped_dir'] = str(self.fg_actors_dir / gait_cfg['ped_dir'])
        self.gait_analyzer = PedGaitMatcher(self.data_dir, gait_cfg)

    def _query_actors_in_view(self, actor_transform: np.ndarray, lidar_pose: np.ndarray) -> bool:
        """Check if an actor's origin is within the occupancy bounding box.

        Args:
            actor_transform: 4x4 actor-to-world transformation matrix
            lidar_pose: 4x4 lidar-to-world transformation matrix

        Returns:
            True if actor origin is within the occupancy bounding box
        """
        actor_world_loc = np.array(actor_transform)[:, 3]
        actor_lidar_loc = (np.linalg.inv(lidar_pose) @ actor_world_loc)[:3]
        return np.all(actor_lidar_loc >= self.occ_bbox[0]) and np.all(actor_lidar_loc <= self.occ_bbox[1])

    def gen_mesh(self, frame_id: int, lidar_pose: np.ndarray, return_panoptic: bool = False) -> Dict:
        """Generate foreground meshes in lidar coordinates.

        Args:
            frame_id: Frame number to process
            lidar_pose: 4x4 lidar-to-world transformation matrix
            return_panoptic: If True, return panoptic format with instance IDs

        Returns:
            If return_panoptic=True:
                {sem_id: List[(instance_id, Trimesh)]} - each element is (instance_id, mesh)
            If return_panoptic=False:
                {sem_id: [mesh1, mesh2, ...]} - list of meshes (semantic format)
        """
        ped_id = name2label['Pedestrian'].id
        fg_meshes = {}

        traffic_info = self.gait_analyzer.load_traffic_info(frame_id)

        # Process vehicles
        for actor in traffic_info['vehicles']:
            if not self._query_actors_in_view(actor['transform'], lidar_pose):
                continue

            # Get semantic class based on vehicle type (Car/Truck/Bus)
            vehicle_semantic_class = get_vehicle_semantic_class(actor['type'])
            vehicle_sem_id = name2label[vehicle_semantic_class].id

            # Initialize list for this semantic class if needed
            if vehicle_sem_id not in fg_meshes:
                fg_meshes[vehicle_sem_id] = []

            vehicle_mesh = self.get_vehicle_mesh(actor, lidar_pose)
            if return_panoptic:
                actor_id = actor.get('id', 0)
                instance_id = get_fg_instance_id(actor_id)
                fg_meshes[vehicle_sem_id].append((instance_id, vehicle_mesh))
            else:
                fg_meshes[vehicle_sem_id].append(vehicle_mesh)

        # Process pedestrians
        for actor in traffic_info['pedestrians']:
            if not self._query_actors_in_view(actor['transform'], lidar_pose):
                continue
            if ped_id not in fg_meshes:
                fg_meshes[ped_id] = []
            ped_mesh = self.get_ped_mesh(actor, frame_id, lidar_pose)
            if return_panoptic:
                instance_id = actor.get('id', 0)
                fg_meshes[ped_id].append((instance_id, ped_mesh))
            else:
                fg_meshes[ped_id].append(ped_mesh)

        return fg_meshes

    def get_vehicle_mesh(self, actor: Dict, lidar_pose: np.ndarray) -> Trimesh:
        """Get vehicle mesh in lidar coordinates.

        Args:
            actor: Actor info dict with 'type' and 'transform'
            lidar_pose: 4x4 lidar-to-world transformation matrix

        Returns:
            Vehicle Trimesh in lidar coordinates
        """
        actor_name = actor['type'].replace('.', '_')
        actor_name = f'{actor_name}.obj'

        # Cache the actor mesh
        if actor_name not in self._cached_actor_meshes:
            vehicle_class = get_vehicle_semantic_class(actor['type'])
            actor_path = self.fg_actors_dir / vehicle_class / actor_name
            self._cached_actor_meshes[actor_name] = self.load_mesh(actor_path)

        # Transform mesh to lidar coordinates
        transformed_mesh = self._cached_actor_meshes[actor_name].copy()
        transformed_mesh.apply_transform(np.array(actor['transform']))  # To world
        transformed_mesh.apply_transform(np.linalg.inv(lidar_pose))  # To lidar
        return transformed_mesh

    def get_ped_mesh(self, actor: Dict, frame_id: int, lidar_pose: np.ndarray) -> Trimesh:
        """Get pedestrian mesh in lidar coordinates with gait animation.

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


class SceneMeshGenerator(BaseMeshGenerator):
    """Combined scene mesh generator (background + foreground)."""

    def __init__(self, cfg: DictConfig, town_name: str, sequence: str):
        super().__init__(cfg, town_name, sequence)
        self.fg_mesh_generator = FgMeshGenerator(cfg, town_name, sequence)
        self.bg_mesh_generator = BgMeshGenerator(cfg, town_name, sequence)

        lidar_pose_path = self.data_dir / "poses" / "lidar.txt"
        self.lidar_poses = {arr[0]: arr[1:].reshape(4, 4) for arr in np.loadtxt(lidar_pose_path)}

    def merge_sem_mesh(self, semantic_meshes: Dict, return_instance: bool = False) -> OrderedDict:
        """Merge semantic meshes in z-order (background to foreground).

        Args:
            semantic_meshes: Dict of {sem_id: [meshes]}
            return_instance: If True, keep instance-level classes as lists

        Returns:
            OrderedDict of merged meshes sorted by z_order
        """
        # Sort semantic IDs by z_order
        sem_ids_sorted = sorted(semantic_meshes.keys(), key=lambda sid: sem_z_order.get(sid, 50))

        merged_sem_meshes = OrderedDict()
        for sem_id in sem_ids_sorted:
            if sem_id in semantic_meshes and len(semantic_meshes[sem_id]) > 0:
                if return_instance and id2label[sem_id].is_instance:
                    # Keep as list for instance classes
                    merged_sem_meshes[sem_id] = semantic_meshes[sem_id]
                else:
                    # Merge meshes for stuff classes
                    merged_mesh = trimesh.util.concatenate(semantic_meshes[sem_id])
                    merged_sem_meshes[sem_id] = merged_mesh
        return merged_sem_meshes

    def generate_sem_mesh(self, frame_id: int) -> OrderedDict:
        """Generate the combined semantic scene mesh for a single frame in lidar coordinates.

        Args:
            frame_id: Frame number to process

        Returns:
            OrderedDict of {sem_id: Trimesh} sorted by z_order
        """
        lidar_pose = self.lidar_poses[frame_id]

        # Generate fg and bg meshes
        fg_mesh = self.fg_mesh_generator.gen_mesh(frame_id, lidar_pose)
        bg_mesh = self.bg_mesh_generator.gen_mesh(lidar_pose)  # {sem_id: [meshes, ...], ...}

        scene_mesh = bg_mesh
        # Merge fg and bg meshes
        for idx in fg_mesh.keys():
            if idx in scene_mesh:
                scene_mesh[idx].extend(fg_mesh[idx])
            else:
                scene_mesh[idx] = fg_mesh[idx]

        # Merge meshes
        scene_mesh = self.merge_sem_mesh(scene_mesh)
        return scene_mesh

    def generate_pano_mesh(self, frame_id: int) -> Dict:
        """Generate panoptic mesh for a single frame in lidar coordinates.

        Args:
            frame_id: Frame number to process

        Returns:
            Panoptic mesh in the format:
            {
                sem_id: List[(instance_id, Trimesh)] for instance classes (is_instance=True)
                sem_id: Trimesh for stuff classes (is_instance=False, instance_id=0)
            }
        """
        lidar_pose = self.lidar_poses[frame_id]

        # Generate fg and bg meshes in panoptic format
        fg_mesh = self.fg_mesh_generator.gen_mesh(frame_id, lidar_pose, return_panoptic=True)
        bg_mesh = self.bg_mesh_generator.gen_mesh(lidar_pose, return_panoptic=True)

        # Merge fg and bg meshes
        pano_mesh = {}

        # Add background meshes
        for sem_id, mesh_data in bg_mesh.items():
            pano_mesh[sem_id] = mesh_data

        # Merge foreground meshes
        for sem_id, fg_instances in fg_mesh.items():
            if len(fg_instances) == 0:
                continue

            if sem_id in pano_mesh:
                # Both bg and fg have this semantic class
                if isinstance(pano_mesh[sem_id], list):
                    # Instance class: directly extend
                    pano_mesh[sem_id].extend(fg_instances)
                else:
                    # bg is stuff but fg is instance: convert to instance format
                    pano_mesh[sem_id] = [(0, pano_mesh[sem_id])] + fg_instances
            else:
                pano_mesh[sem_id] = fg_instances

        return pano_mesh


class BgSemanticSceneGenerator(BaseMeshGenerator):
    """Generate semantic mesh of the background scene with specific semantics."""

    def __init__(self,
                 cfg: DictConfig,
                 town_name: str,
                 sequence: str,
                 carla_class: List[str] = None,
                 semantic_class: List[str] = None
                 ):
        super().__init__(cfg, town_name, sequence)

        self.carla_class = carla_class if carla_class is not None else cfg.get('sem_depth_correction', {}).get(
            'carla_class', [])
        self.semantic_class = semantic_class if semantic_class is not None else cfg.get('sem_depth_correction', {}).get(
            'semantic_class', [])

        self.bg_mesh_generator = BgMeshGenerator(cfg, town_name, sequence)
        self.bg_scene = None
        self.bg_sem_scene = None
        self._init_scene()

    def _init_scene(self) -> None:
        """Initialize background semantic scene from actor meshes."""
        actors = [actor for actor in self.bg_mesh_generator.actors if
                  actor['ue_class'] in self.carla_class or actor['semantic_label'] in self.semantic_class]

        sem_scene = {}
        for actor in actors:
            sem_id = name2label[actor['semantic_label']].id
            if sem_id not in sem_scene:
                sem_scene[sem_id] = []
            mesh = self.bg_mesh_generator.load_mesh(actor)
            sem_scene[sem_id].append(mesh)
        self.bg_sem_scene = {key: trimesh.util.concatenate(mesh_list) for key, mesh_list in sem_scene.items()}
        self.bg_scene = trimesh.util.concatenate(self.bg_sem_scene.values())

    def get_mesh(self) -> Trimesh:
        """Get combined background scene mesh."""
        return self.bg_scene

    def get_sem_mesh(self) -> Dict[int, Trimesh]:
        """Get semantic scene mesh dictionary.

        Returns:
            Dict mapping semantic IDs to Trimesh objects (read-only after init)
        """
        return self.bg_sem_scene


class FgVehicleMeshGenerator(BaseMeshGenerator):
    """Generate meshes for all moving vehicles in the scene."""

    def __init__(self, cfg: DictConfig, town_name: str, sequence: str):
        super().__init__(cfg, town_name, sequence)
        self.fg_actors_dir = self.scene_mesh_dir / "fg_actors"
        self._cached_actor_meshes = {}  # Cache for loaded actor meshes

    def get_vehicle_mesh(self, actor: Dict) -> Trimesh:
        """Get vehicle mesh in world coordinate system.

        Args:
            actor: Actor info dict with 'type' and 'transform'

        Returns:
            Vehicle Trimesh in world coordinates
        """
        actor_name = actor['type'].replace('.', '_')
        actor_name = f'{actor_name}.obj'

        # Cache the actor mesh
        if actor_name not in self._cached_actor_meshes:
            vehicle_class = get_vehicle_semantic_class(actor['type'])
            actor_path = self.fg_actors_dir / vehicle_class / actor_name
            self._cached_actor_meshes[actor_name] = self.load_mesh(actor_path)

        # Transform mesh to world coordinates
        transformed_mesh = self._cached_actor_meshes[actor_name].copy()
        transformed_mesh.apply_transform(np.array(actor['transform']))  # To world
        return transformed_mesh

    def get_sem_mesh(self, frame_id: int) -> Dict[str, Trimesh]:
        """Get semantic vehicle meshes for a frame.

        Args:
            frame_id: Frame number

        Returns:
            Dict mapping vehicle class name to merged Trimesh
        """
        traffic_info = self.load_traffic_info(frame_id)

        vehicle_sem_meshes = {}
        for actor in traffic_info['vehicles']:
            vehicle_mesh = self.get_vehicle_mesh(actor)
            vehicle_class = get_vehicle_semantic_class(actor['type'])
            if vehicle_class not in vehicle_sem_meshes:
                vehicle_sem_meshes[vehicle_class] = []
            vehicle_sem_meshes[vehicle_class].append(vehicle_mesh)

        # Combine meshes of the same type
        vehicle_sem_meshes = {k: trimesh.util.concatenate(v) for k, v in vehicle_sem_meshes.items()}
        return vehicle_sem_meshes

    def get_mesh(self, frame_id: int) -> Trimesh:
        """Get combined vehicle mesh for a frame.

        Args:
            frame_id: Frame number

        Returns:
            Merged Trimesh of all vehicles in world coordinates
        """
        traffic_info = self.load_traffic_info(frame_id)

        vehicle_meshes = []
        for actor in traffic_info['vehicles']:
            vehicle_mesh = self.get_vehicle_mesh(actor)
            vehicle_meshes.append(vehicle_mesh)

        vehicle_mesh = trimesh.util.concatenate(vehicle_meshes)
        return vehicle_mesh


class SemanticSceneGenerator(BaseMeshGenerator):
    """Generate semantic mesh of the full scene (background + foreground vehicles)."""

    def __init__(self,
                 cfg: DictConfig,
                 town_name: str,
                 sequence: str,
                 carla_class: List[str] = None,
                 semantic_class: List[str] = None
                 ):
        super().__init__(cfg, town_name, sequence)

        self.bg_sem_scene_generator = BgSemanticSceneGenerator(cfg, town_name, sequence, carla_class, semantic_class)
        self.fg_vehicle_generator = FgVehicleMeshGenerator(cfg, town_name, sequence)

    def get_mesh(self, frame_id: int) -> Trimesh:
        """Get combined scene mesh for a frame.

        Args:
            frame_id: Frame number

        Returns:
            Merged Trimesh of background + foreground in world coordinates
        """
        bg_scene = self.bg_sem_scene_generator.get_mesh()
        fg_vehicle = self.fg_vehicle_generator.get_mesh(frame_id)
        scene_mesh = trimesh.util.concatenate([bg_scene, fg_vehicle])
        return scene_mesh

    def get_sem_mesh(self, frame_id: int) -> Dict[int, Trimesh]:
        """Get semantic scene mesh for a frame.

        Args:
            frame_id: Frame number

        Returns:
            Dict mapping semantic IDs to Trimesh objects
        """
        # Get background semantic meshes (already in world coordinates)
        sem_mesh_dict = {}
        for sem_id, mesh in self.bg_sem_scene_generator.bg_sem_scene.items():
            sem_mesh_dict[sem_id] = mesh

        # Get foreground vehicle semantic meshes (already in world coordinates)
        fg_sem_meshes = self.fg_vehicle_generator.get_sem_mesh(frame_id)

        # Merge foreground into background
        for sem_label, fg_mesh in fg_sem_meshes.items():
            sem_id = name2label[sem_label].id
            if sem_id in sem_mesh_dict:
                # Concatenate meshes of the same semantic class
                bg_mesh = sem_mesh_dict[sem_id]
                combined_vertices = np.vstack([bg_mesh.vertices, fg_mesh.vertices])
                offset_faces = fg_mesh.faces + len(bg_mesh.vertices)
                combined_faces = np.vstack([bg_mesh.faces, offset_faces])
                sem_mesh_dict[sem_id] = trimesh.Trimesh(
                    vertices=combined_vertices,
                    faces=combined_faces,
                    process=False
                )
            else:
                sem_mesh_dict[sem_id] = fg_mesh

        return sem_mesh_dict
