from pathlib import Path
from typing import Dict, Tuple

import cv2
import hydra
import numpy as np
import open3d as o3d
import open3d.core as o3c
import yaml
from omegaconf import DictConfig
from tqdm import tqdm
from trimesh import Trimesh

from occupancy_generation.generators.mesh_generator import SemanticSceneGenerator
from utils.labels import id2color, name2label
from utils.ray_casting import RayCasting
from utils.transforms import transform_mat_world_to_o3d
from utils.vis_utils import vis_semantics, vis_depth


class SemDepthGenerator:
    """Generate refined semantic and depth maps via mesh ray casting."""

    def __init__(self, cfg: DictConfig, town_name: str, sequence: str):
        self.cfg = cfg
        dataset_dir = Path(cfg.dataset_dir)
        self.data_dir = dataset_dir / f"{town_name}_Seq{sequence}"

        # Load camera and LiDAR data
        lidar_pose_path = self.data_dir / "poses" / "lidar.txt"
        self.lidar_poses = {arr[0]: arr[1:].reshape(4, 4) for arr in np.loadtxt(lidar_pose_path)}
        with open(dataset_dir / "calib" / "calib.yaml", 'r') as f:
            self.calib = yaml.safe_load(f)
        self.cam_to_lidar = {
            f"0{i}": np.array(self.calib['sensors'][f'cam_0{i}']['transform'])
            for i in range(6)
        }

        # Initialize single ray caster
        self.ray_caster = RayCasting(
            width=cfg.image_width,
            height=cfg.image_height
        )

        # Camera intrinsic (shared by all cameras)
        self.intrinsic = np.array(cfg.intrinsic_matrix)
        self.intrinsic_tensor = o3c.Tensor(self.intrinsic.astype(np.float32), dtype=o3c.Dtype.Float32)

        self.lut_rgb2sem = np.zeros(256 ** 3, dtype=np.int32)
        for sem_id, (r, g, b) in id2color.items():
            self.lut_rgb2sem[(r << 16) + (g << 8) + b] = sem_id

    def generate_sem_depth(
            self,
            sem_mesh_dict: Dict[int, Trimesh],
            frame_id: int
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generate refined depth and semantic maps for all cameras.

        Args:
            sem_mesh_dict: Dict mapping semantic IDs to Trimesh objects
            frame_id: Frame ID

        Returns:
            Dict mapping camera ID to (refined_depth, refined_semantic) tuples
        """
        # Filter valid meshes
        valid_meshes = {}
        for sem_id, mesh in sem_mesh_dict.items():
            if mesh is not None and len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                valid_meshes[sem_id] = mesh

        if not valid_meshes:
            print(f"Warning: No valid semantic meshes for frame {frame_id}")
            return {}

        # Build scene once for all cameras
        scene = o3d.t.geometry.RaycastingScene()
        mesh2sem = {}  # {mesh_id: sem_id}

        for sem_id, mesh in valid_meshes.items():
            # Create Open3D mesh (in world coordinates)
            o3d_mesh = self.ray_caster.create_triangle_mesh(mesh.vertices, mesh.faces)

            # Add to scene and record mapping
            mesh_id = scene.add_triangles(o3d_mesh)
            mesh2sem[mesh_id] = sem_id

        results = {}
        for cam_id, cam_transform in self.cam_to_lidar.items():
            # Load original depth map and semantic map
            collected_depth_path = self.data_dir / "depth_carla" / f'image_{cam_id}' / f"{frame_id:04d}.png"
            collected_depth = cv2.imread(str(collected_depth_path), -1)
            collected_depth = collected_depth.astype(np.float32) / 65535 * 80.0

            collected_semantic_path = self.data_dir / "semantics_carla" / f'image_{cam_id}' / f"{frame_id:04d}.png"
            collected_semantic_rgb = cv2.imread(str(collected_semantic_path), cv2.IMREAD_UNCHANGED)
            collected_semantic_rgb = cv2.cvtColor(collected_semantic_rgb, cv2.COLOR_BGR2RGB)

            # Convert RGB color map to semantic ID map
            h, w = collected_depth.shape

            flat_key = (collected_semantic_rgb[:, :, 0].astype(np.int32) << 16) | \
                       (collected_semantic_rgb[:, :, 1].astype(np.int32) << 8) | \
                       collected_semantic_rgb[:, :, 2].astype(np.int32)

            collected_semantic = self.lut_rgb2sem[flat_key]

            # Execute ray casting
            rays = self.create_rays(frame_id, cam_transform)
            geometry_ids, raytraced_depth = self.ray_caster.cast_rays_multiple_meshes(scene, rays)

            # Generate raytraced semantic map
            raytraced_semantic = np.zeros((h, w), dtype=np.int32)
            for mesh_id, sem_id in mesh2sem.items():
                raytraced_semantic[geometry_ids == mesh_id] = sem_id

            # Generate refined depth map
            refined_depth = np.minimum(collected_depth, raytraced_depth)

            # Refine semantic map
            refined_semantic = collected_semantic.copy()
            refined_semantic[collected_semantic == name2label['None'].id] = name2label['Terrain'].id

            INVALID_ID = 4294967295  # Open3D's invalid geometry id
            valid_mask = ((geometry_ids != INVALID_ID)
                          & (raytraced_depth - refined_depth < 0.1))
            refined_semantic[valid_mask] = raytraced_semantic[valid_mask]

            results[cam_id] = (refined_depth, refined_semantic)

        return results

    def create_rays(self, frame_id: int, cam_transform: np.ndarray):
        """Create rays for a camera at a specific frame.

        Args:
            frame_id: Frame number
            cam_transform: Camera-to-lidar transformation matrix

        Returns:
            Open3D ray tensor
        """
        # Calculate camera pose in world coordinate system
        cam_pose = self.lidar_poses[frame_id] @ cam_transform
        extrinsic = transform_mat_world_to_o3d @ np.linalg.inv(cam_pose)

        extrinsic_tensor = o3c.Tensor(extrinsic.astype(np.float32), dtype=o3c.Dtype.Float32)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=self.intrinsic_tensor,
            extrinsic_matrix=extrinsic_tensor,
            width_px=self.ray_caster.width,
            height_px=self.ray_caster.height,
        )
        return rays


@hydra.main(config_path="../config", config_name="gen_sem_depth", version_base=None)
def main(cfg: DictConfig) -> None:
    for town_name in cfg.town_names:
        for sequence in cfg.sequences:
            print(f"\nGenerating semantic and depth: {town_name} Seq{sequence}")
            generator = SemDepthGenerator(cfg, town_name, sequence)
            mesh_generator = SemanticSceneGenerator(
                cfg, town_name, sequence,
                carla_class=list(cfg.sem_depth_correction.carla_class),
                semantic_class=list(cfg.sem_depth_correction.semantic_class),
            )

            for i in tqdm(range(cfg.frame_range[0], cfg.frame_range[1] + 1),
                          desc=f"{town_name}_Seq{sequence}"):
                sem_mesh = mesh_generator.get_sem_mesh(i)
                results = generator.generate_sem_depth(sem_mesh, i)

                for cam_id, (refined_depth, refined_semantic) in results.items():
                    # Save refined depth map
                    depth_path = generator.data_dir / "depth" / f'image_{cam_id}' / f"{i:04d}.png"
                    depth_path.parent.mkdir(parents=True, exist_ok=True)
                    vis_depth(refined_depth, save_path=depth_path)

                    # Save refined semantic map
                    sem_path = generator.data_dir / "semantics" / f'image_{cam_id}' / f"{i:04d}.png"
                    sem_path.parent.mkdir(parents=True, exist_ok=True)
                    vis_semantics(refined_semantic, save_path=sem_path)


if __name__ == '__main__':
    main()
