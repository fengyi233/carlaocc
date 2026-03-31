import os
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict

import cv2
import numpy as np
import open3d as o3d
import trimesh
from matplotlib import pyplot as plt

from utils.data_converter import decode_combined_id
from utils.labels import (
    id2color,
    nuscenes_id2color,
    kitti360_id2color,
    color_LUT,
)
from utils.occ_ops import occ_dense2sparse
from utils.transforms import transform_mat_world_to_o3d

rng = np.random.default_rng(seed=42)
INSTANCE_COLOR_OFFSET_LUT = 0.3 * (rng.random(size=(2048, 3)) * 2 - 1)
INSTANCE_COLOR_OFFSET_LUT[0] = (0, 0, 0)


def get_color_from_colormap(sem_id: int, colormap: str = 'carla') -> tuple:
    """
    Get RGB color for a semantic ID based on the specified colormap.
    
    Args:
        sem_id: Semantic label ID
        colormap: One of 'carla', 'nuscenes', 'kitti360'. Defaults to 'carla'.
    
    Returns:
        RGB color tuple (0-255 range)
    """
    if colormap == 'carla':
        return id2color.get(sem_id, (0, 0, 0))
    elif colormap == 'nuscenes':
        return nuscenes_id2color.get(sem_id, (0, 0, 0))
    elif colormap == 'kitti360':
        return kitti360_id2color.get(sem_id, (0, 0, 0))
    else:
        raise ValueError(f"Unsupported colormap: {colormap}. Choose from 'carla', 'nuscenes', 'kitti360'.")


class CameraSettingConfig:
    def __init__(
            self,
            image_size: Tuple[int, int] = (1920, 1080),
            focal_length: float = 1000.0,
            cam_position: Tuple[float, float, float] = (0., 0., 0.),
            cam_pitch: float = 0.,
    ):
        """
        Camera setting configuration class.

        Args:
            image_size: Image size as (width, height)
            focal_length: Focal length
            cam_position: Camera position (x, y, z) in world coordinates
            cam_pitch: Camera pitch angle in degrees
        """
        self.image_size = image_size
        # World coordinate: x-forward, y-left, z-up
        # Open3D camera: x-right, y-down, z-forward
        self.extrinsic = transform_mat_world_to_o3d.astype(np.float32).copy()
        # world --> cam
        rel_pos = -1 * np.array([
            -cam_position[1],
            -cam_position[2],
            cam_position[0]
        ])
        self.extrinsic[:3, 3] = rel_pos

        cam_pitch = np.radians(cam_pitch)
        pitch_mat = np.array([
            [1, 0, 0, 0],
            [0, np.cos(cam_pitch), -np.sin(cam_pitch), 0],
            [0, np.sin(cam_pitch), np.cos(cam_pitch), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.extrinsic = pitch_mat @ self.extrinsic

        self.intrinsic = np.array([
            [focal_length, 0.0, image_size[0] / 2],
            [0.0, focal_length, image_size[1] / 2],
            [0.0, 0.0, 1.0]
        ])


class CameraSetting:
    """
    Camera settings class for Open3D visualization with predefined camera configurations.
    
    This class encapsulates camera parameters including extrinsic and intrinsic matrices,
    image size, and provides convenient access to different camera viewpoints.
    """

    def __init__(self,
                 extrinsic: np.ndarray,
                 intrinsic: np.ndarray,
                 image_size: Tuple[int, int]):
        """
        Initialize camera setting with extrinsic and intrinsic parameters.
        
        Args:
            extrinsic: 4x4 extrinsic transformation matrix
            intrinsic: 3x3 intrinsic camera matrix
            image_size: Image dimensions as (width, height)
        """
        assert extrinsic.shape == (4, 4), f"Extrinsic matrix must be 4x4, got {extrinsic.shape}"
        assert intrinsic.shape == (3, 3), f"Intrinsic matrix must be 3x3, got {intrinsic.shape}"
        assert len(image_size) == 2, f"Image size must be (width, height), got {image_size}"

        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.image_size = image_size

    @classmethod
    def from_config(cls, config: CameraSettingConfig) -> 'CameraSetting':
        return cls(config.extrinsic, config.intrinsic, config.image_size)

    @classmethod
    def default_vis_mesh(cls) -> 'CameraSetting':
        # 30m higher, 25.6m backward, 45 deg forward down
        config = CameraSettingConfig(
            image_size=(1920, 1080),
            cam_position=(-25.6, 0., 30.),
            cam_pitch=45
        )
        return cls.from_config(config)

    @classmethod
    def default_vis_occ(cls) -> 'CameraSetting':
        config = CameraSettingConfig(
            cam_position=(0, 25.6, 30),
            cam_pitch=45
        )
        return cls.from_config(config)

    @classmethod
    def default_vis_lidar(cls) -> 'CameraSetting':
        config = CameraSettingConfig(
            image_size=(1000, 1000),
            focal_length=1000.0,
            cam_position=(0., 0., 50.),
            cam_pitch=90,
        )
        return cls.from_config(config)

    @classmethod
    def default_setting(cls) -> 'CameraSetting':
        config = CameraSettingConfig()
        return cls.from_config(config)

    def get_pinhole_camera_parameters(self) -> o3d.camera.PinholeCameraParameters:
        """
        Convert camera setting to Open3D PinholeCameraParameters.
        
        Returns:
            Open3D PinholeCameraParameters object
        """
        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.extrinsic = self.extrinsic
        camera_params.intrinsic.intrinsic_matrix = self.intrinsic
        camera_params.intrinsic.width = self.image_size[0]
        camera_params.intrinsic.height = self.image_size[1]
        return camera_params


def vis_depth(depth: np.ndarray,
              max_depth: float = 80.0,
              save_path: Optional[Union[str, Path]] = None) -> None:
    depth = np.clip(depth, 0, max_depth)
    if save_path is not None:
        depth = (65535.0 / max_depth * depth).astype(np.uint16)
        cv2.imwrite(str(save_path), depth)
    else:
        plt.figure('Depth')
        plt.imshow(depth, cmap='plasma', vmin=0, vmax=max_depth)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()


def vis_semantics(sem_img: np.ndarray,
                  color_map: Dict = None,
                  save_path: Optional[Union[str, Path]] = None) -> None:
    if color_map is None:
        from utils.labels import labels
        max_id = max(lb.id for lb in labels)
        color_map = np.zeros((max_id + 1, 3), dtype=np.uint8)
        for lb in labels:
            color_map[lb.id] = lb.color

    color_img = color_map[sem_img]

    if save_path is not None:
        cv2.imwrite(str(save_path), cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
    else:
        plt.figure('Semantic Segmentation')
        plt.imshow(color_img)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()


def vis_pc(pc: np.ndarray,
           pc_type: str = 'semantic_lidar',
           camera_setting: Optional[CameraSetting] = CameraSetting.default_vis_lidar(),
           save_path: Optional[Union[str, Path]] = None) -> None:
    assert len(pc.shape) == 2 and pc.shape[1] >= 3, "Input must be 2D array with at least XYZ coordinates"

    # Create point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc[:, :3])

    # Set colors based on mode
    if pc_type == 'semantic_lidar' and pc.shape[1] == 6:
        CosAngle = pc[:, 3:4]
        ObjIdx = pc[:, 4:5]
        ObjTag = pc[:, 5:].astype(np.int32).flatten()
        colors = color_LUT[ObjTag]

    elif pc_type == 'lidar' and pc.shape[1] == 4:
        intensity = pc[:, 3:4]
        # normalize intensity
        intensity = intensity.astype(np.float32)
        min_v = intensity.min()
        max_v = intensity.max()
        normed = 1.0 - (intensity - min_v) / (max_v - min_v)
        colors = np.repeat(normed.reshape(-1, 1), 3, axis=1)
    else:
        raise ValueError('Unknown point cloud type {}'.format(pc_type))
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Setup visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=camera_setting.image_size[0],
                      height=camera_setting.image_size[1],
                      visible=(save_path is None))
    vis.add_geometry(point_cloud)
    render_opt = vis.get_render_option()
    render_opt.point_size = 3.0
    # Configure viewpoint
    ctr = vis.get_view_control()
    camera_params = camera_setting.get_pinhole_camera_parameters()
    ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    # Display or save visualization
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis.capture_screen_image(save_path, do_render=True)
    else:
        vis.run()
    vis.destroy_window()


def vis_occ(occ: np.ndarray,
            voxel_size: float = 0.2,
            bbox: Optional[np.ndarray] = None,
            camera_setting: Optional[CameraSetting] = CameraSetting.default_vis_occ(),
            show_axis: bool = True,
            save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Visualize 3D occupancy grid with Open3D.

    Args:
        occ: 3D occupancy grid array (1=occupied, 0=empty)
        voxel_size: Size of each voxel in meters
        bbox: Optional bounding box to visualize (shape: (2, 3) for min/max bounds)
        camera_setting: Optional CameraSetting object for predefined camera configurations
        show_axis: Whether to display the coordinate frame
        save_path: Optional path to save visualization image
    """
    assert occ.ndim == 3 or (occ.ndim == 2 and occ.shape[1] in [3, 4]), \
        "Occupancy grid must be in shape of XxYxZ, Nx3, or Nx4"
    if occ.ndim == 3:
        occ = occ_dense2sparse(occ)

    # Use raw indices as coordinates (scaled by voxel size for proper proportions)
    points = occ[:, :3] * voxel_size

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Color points by height (Z coordinate)
    z = occ[:, 2]
    z_normalized = (z - z.min()) / (z.max() - z.min() + 1e-6)  # Avoid division by zero
    colors = plt.cm.viridis(z_normalized)[:, :3]  # Get RGB colors from colormap
    pcd.colors = o3d.utility.Vector3dVector(colors)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd,
        voxel_size=voxel_size,
    )
    # Setup visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=camera_setting.image_size[0],
                      height=camera_setting.image_size[1],
                      visible=(save_path is None))
    vis.add_geometry(voxel_grid)

    # Add coordinate frame
    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        vis.add_geometry(axis)

    # Add bounding box
    if bbox is not None:
        assert bbox.shape == (2, 3), f"Bbox must have shape (2, 3), got {bbox.shape}"
        min_bound, max_bound = bbox
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
        bbox_lines.paint_uniform_color([1.0, 0.0, 0.0])  # red
        vis.add_geometry(bbox_lines)

    # Configure viewpoint
    ctr = vis.get_view_control()
    camera_params = camera_setting.get_pinhole_camera_parameters()
    ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    # Display or save visualization
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis.capture_screen_image(save_path, do_render=True)
    else:
        vis.run()
    vis.destroy_window()


def vis_sem_occ(occ: np.ndarray,
                voxel_size: float = 0.2,
                bbox: Optional[np.ndarray] = None,
                camera_setting: Optional[CameraSetting] = CameraSetting.default_vis_occ(),
                show_axis: bool = True,
                colormap: str = 'carla',
                save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Visualize 3D semantic occupancy grid using Open3D.

    Args:
        occ: 3D occupancy grid array with semantic labels (integer IDs).
        voxel_size: Size of each voxel in meters.
        bbox: Optional bounding box to visualize (shape: (2, 3) for min/max bounds).
        camera_setting: Optional CameraSetting object for predefined camera configurations.
        show_axis: Whether to display the coordinate frame.
        colormap: Color mapping scheme. One of 'carla', 'nuscenes', 'kitti360'. Defaults to 'carla'.
        save_path: Optional path to save visualization as image.
    """
    assert occ.ndim == 3 or (occ.ndim == 2 and occ.shape[1] in [3, 4]), \
        "Occupancy grid must be in shape of XxYxZ, Nx3, or Nx4"
    if occ.ndim == 3:
        occ = occ_dense2sparse(occ)

    points = occ[:, :3] * voxel_size
    sem_ids = occ[:, 3]

    # Map semantic IDs to RGB colors and normalize to [0,1]
    rgb_colors = np.array([get_color_from_colormap(int(sem_id), colormap) for sem_id in sem_ids]) / 255.0

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors)

    # Convert point cloud to voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    # Visualizer setup
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=camera_setting.image_size[0],
                      height=camera_setting.image_size[1],
                      visible=(save_path is None))
    vis.add_geometry(voxel_grid)

    # Add coordinate frame
    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        vis.add_geometry(axis)

    # Add bounding box
    if bbox is not None:
        assert bbox.shape == (2, 3), f"Bbox must have shape (2, 3), got {bbox.shape}"
        min_bound, max_bound = bbox
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
        bbox_lines.paint_uniform_color([1.0, 0.0, 0.0])  # red
        vis.add_geometry(bbox_lines)

    ctr = vis.get_view_control()
    camera_params = camera_setting.get_pinhole_camera_parameters()
    ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    # Display or save visualization
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis.capture_screen_image(save_path, do_render=True)
    else:
        vis.run()
    vis.destroy_window()


def vis_mesh(mesh: Union[trimesh.Trimesh, o3d.geometry.TriangleMesh],
             bbox: Optional[np.ndarray] = None,
             camera_setting: Optional[CameraSetting] = CameraSetting.default_vis_mesh(),
             show_axis: bool = True,
             save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Visualize 3D mesh using Open3D.

    Args:
        mesh: Input mesh to visualize
        bbox: Optional bounding box to visualize (shape: (2, 3) for min/max bounds)
        camera_setting: Optional CameraSetting object for predefined camera configurations
        show_axis: Whether to display the coordinate frame
        save_path: Optional path to save the visualization as an image
    """

    if isinstance(mesh, trimesh.Trimesh):
        # Convert to Open3D format
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()
        o3d_mesh.paint_uniform_color((0.2, 0.6, 0.8))
    elif isinstance(mesh, o3d.geometry.TriangleMesh):
        o3d_mesh = mesh
    else:
        raise TypeError(f"Unsupported mesh type: {type(mesh)}. ")
    # Setup visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=camera_setting.image_size[0],
                      height=camera_setting.image_size[1],
                      visible=(save_path is None))
    vis.add_geometry(o3d_mesh)

    # Add coordinate frame
    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        vis.add_geometry(axis)

    # Add bounding box
    if bbox is not None:
        assert bbox.shape == (2, 3), f"Bbox must have shape (2, 3), got {bbox.shape}"
        min_bound, max_bound = bbox
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
        bbox_lines.paint_uniform_color([1.0, 0.0, 0.0])  # red
        vis.add_geometry(bbox_lines)

    # Configure viewpoint
    ctr = vis.get_view_control()
    camera_params = camera_setting.get_pinhole_camera_parameters()
    ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    # Configure rendering options
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True  # Enable back face rendering
    render_option.mesh_show_wireframe = False

    # Display or save visualization
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis.capture_screen_image(save_path, do_render=True)
    else:
        vis.run()
    vis.destroy_window()


def vis_sem_mesh(sem_mesh: Dict[int, trimesh.Trimesh],
                 bbox: Optional[np.ndarray] = None,
                 camera_setting: Optional[CameraSetting] = CameraSetting.default_vis_mesh(),
                 show_axis: bool = True,
                 colormap: str = 'carla',
                 save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Visualize 3D semantic meshes using Open3D with consistent styling.
    
    This function creates an interactive 3D visualization of semantic meshes where each
    semantic class is colored according to its predefined color in the label mapping.
    
    Args:
        sem_mesh: Dictionary mapping semantic IDs to trimesh.Trimesh objects
        bbox: Optional bounding box to visualize (shape: (2, 3) for min/max bounds)
        camera_setting: Optional CameraSetting object for predefined camera configurations
        show_axis: Whether to display the coordinate frame
        colormap: Color mapping scheme. One of 'carla', 'nuscenes', 'kitti360'. Defaults to 'carla'.
        save_path: Optional path to save the visualization as an image
    """
    # Setup visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=camera_setting.image_size[0],
                      height=camera_setting.image_size[1],
                      visible=(save_path is None))

    # Convert each semantic mesh to Open3D format
    for sem_id, mesh in sem_mesh.items():
        if mesh is None:
            continue
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()
        # Apply semantic color
        color = get_color_from_colormap(sem_id, colormap)
        o3d_mesh.paint_uniform_color([c / 255.0 for c in color])
        vis.add_geometry(o3d_mesh)

    # Add coordinate frame
    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        vis.add_geometry(axis)

    # Add bounding box
    if bbox is not None:
        assert bbox.shape == (2, 3), f"Bbox must have shape (2, 3), got {bbox.shape}"
        min_bound, max_bound = bbox
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
        bbox_lines.paint_uniform_color([1.0, 0.0, 0.0])  # red
        vis.add_geometry(bbox_lines)

    # Apply camera setting
    ctr = vis.get_view_control()
    camera_params = camera_setting.get_pinhole_camera_parameters()
    ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    # Configure rendering options
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True  # Enable back face rendering
    render_option.mesh_show_wireframe = False

    # Display or save visualization
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis.capture_screen_image(save_path, do_render=True)
    else:
        vis.run()
    vis.destroy_window()


def vis_sem_mesh_wireframe(sem_mesh: Dict[int, trimesh.Trimesh],
                           bbox: Optional[np.ndarray] = None,
                           camera_setting: Optional[CameraSetting] = CameraSetting.default_setting(),
                           show_axis: bool = False,
                           colormap: str = 'carla',
                           save_path: Optional[Union[str, Path]] = None,
                           line_width: float = 1.0) -> None:
    """
    Visualize 3D semantic meshes as wireframe using Open3D.
    
    This function creates an interactive 3D visualization of semantic meshes where each
    semantic class is displayed as wireframe edges colored according to its predefined
    color in the label mapping.
    
    Args:
        sem_mesh: Dictionary mapping semantic IDs to trimesh.Trimesh objects
        bbox: Optional bounding box to visualize (shape: (2, 3) for min/max bounds)
        camera_setting: Optional CameraSetting object for predefined camera configurations
        show_axis: Whether to display the coordinate frame
        colormap: Color mapping scheme. One of 'carla', 'nuscenes', 'kitti360'. Defaults to 'carla'.
        save_path: Optional path to save the visualization as an image
        line_width: Width of wireframe lines
    """
    # Setup visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=camera_setting.image_size[0],
                      height=camera_setting.image_size[1],
                      visible=(save_path is None))

    # Convert each semantic mesh to Open3D LineSet format
    for sem_id, mesh in sem_mesh.items():
        if mesh is None:
            continue

        # Get semantic color
        color = np.array(get_color_from_colormap(sem_id, colormap)) / 255.0

        # Create Open3D mesh first
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

        # Create LineSet from mesh edges
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)

        # Color all edges with semantic color
        num_lines = len(line_set.lines)
        colors = np.tile(color, (num_lines, 1))
        line_set.colors = o3d.utility.Vector3dVector(colors)

        vis.add_geometry(line_set)

    # Add coordinate frame
    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        vis.add_geometry(axis)

    # Add bounding box
    if bbox is not None:
        assert bbox.shape == (2, 3), f"Bbox must have shape (2, 3), got {bbox.shape}"
        min_bound, max_bound = bbox
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
        bbox_lines.paint_uniform_color([1.0, 0.0, 0.0])  # red
        vis.add_geometry(bbox_lines)

    # Configure camera parameters
    ctr = vis.get_view_control()
    camera_params = camera_setting.get_pinhole_camera_parameters()
    ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    # Configure rendering options
    render_option = vis.get_render_option()
    render_option.line_width = line_width

    # Display or save visualization
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis.capture_screen_image(save_path, do_render=True)
    else:
        vis.run()
    vis.destroy_window()


def vis_pano_mesh(pano_mesh: Dict[int, Union[trimesh.Trimesh, List[tuple]]],
                  bbox: Optional[np.ndarray] = None,
                  camera_setting: Optional[CameraSetting] = CameraSetting.default_vis_mesh(),
                  show_axis: bool = True,
                  colormap: str = 'carla',
                  save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Visualize 3D panoptic meshes using Open3D with COCO-style instance coloring.
    
    This function creates an interactive 3D visualization of panoptic meshes following
    COCO panoptic segmentation visualization convention:
    - Instance classes (instance_id > 0): Each instance gets a unique deterministic random color
      based on its global instance_id (not semantic color)
    - Stuff classes (instance_id = 0): Use uniform semantic colors
    
    Args:
        pano_mesh: Dictionary with two possible value types:
            - For instance classes (is_instance=True): {sem_id: [(instance_id, mesh), ...]}
              List of (instance_id, mesh) tuples
            - For stuff classes (is_instance=False): {sem_id: mesh}
              Single merged mesh with instance_id=0
        bbox: Optional bounding box to visualize (shape: (2, 3) for min/max bounds)
        camera_setting: Optional CameraSetting object for predefined camera configurations
        show_axis: Whether to display the coordinate frame
        colormap: Color mapping scheme. One of 'carla', 'nuscenes', 'kitti360'. Defaults to 'carla'.
        save_path: Optional path to save the visualization as an image
        
    """
    # Setup visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=camera_setting.image_size[0],
                      height=camera_setting.image_size[1],
                      visible=(save_path is None))

    # Process each semantic class
    for sem_id, mesh_data in pano_mesh.items():
        if mesh_data is None:
            continue

        # Get base semantic color (for stuff classes)
        base_color = np.array(get_color_from_colormap(sem_id, colormap)) / 255.0

        # Check if mesh_data is a list (instance class) or single mesh (stuff class)
        if isinstance(mesh_data, list):
            # Instance-level class: color each instance with deterministic random color
            # Following COCO convention: use global instance_id for color generation
            for instance_id, mesh in mesh_data:
                if mesh is None or len(mesh.vertices) == 0:
                    continue
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
                o3d_mesh.compute_vertex_normals()
                instance_color = np.clip(
                    base_color + INSTANCE_COLOR_OFFSET_LUT[instance_id % len(INSTANCE_COLOR_OFFSET_LUT)],
                    0, 1)
                o3d_mesh.paint_uniform_color(instance_color)
                vis.add_geometry(o3d_mesh)
        else:
            # Stuff class: single mesh with uniform semantic color
            mesh = mesh_data
            if mesh is None or len(mesh.vertices) == 0:
                continue
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            o3d_mesh.compute_vertex_normals()
            o3d_mesh.paint_uniform_color(base_color)
            vis.add_geometry(o3d_mesh)

    # Add coordinate frame
    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        vis.add_geometry(axis)

    # Add bounding box
    if bbox is not None:
        assert bbox.shape == (2, 3), f"Bbox must have shape (2, 3), got {bbox.shape}"
        min_bound, max_bound = bbox
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
        bbox_lines.paint_uniform_color([1.0, 0.0, 0.0])  # red
        vis.add_geometry(bbox_lines)

    # Configure camera parameters
    ctr = vis.get_view_control()
    camera_params = camera_setting.get_pinhole_camera_parameters()
    ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    # Configure rendering options
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True  # Enable back face rendering
    render_option.mesh_show_wireframe = False
    if bbox is not None:
        render_option.line_width = 5.0

    # Display or save visualization
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis.capture_screen_image(save_path, do_render=True)
    else:
        vis.run()
    vis.destroy_window()


def vis_pano_mesh_wireframe(pano_mesh: Dict[int, Union[trimesh.Trimesh, List[tuple]]],
                            bbox: Optional[np.ndarray] = None,
                            camera_setting: Optional[CameraSetting] = CameraSetting.default_setting(),
                            show_axis: bool = False,
                            colormap: str = 'carla',
                            save_path: Optional[Union[str, Path]] = None,
                            line_width: float = 1.0) -> None:
    """
    Visualize 3D panoptic meshes as wireframe using Open3D with COCO-style instance coloring.
    
    This function creates an interactive 3D visualization of panoptic meshes as wireframe
    following COCO panoptic segmentation visualization convention:
    - Instance classes (instance_id > 0): Each instance gets a unique deterministic random color
      based on its global instance_id
    - Stuff classes (instance_id = 0): Use uniform semantic colors
    
    Args:
        pano_mesh: Dictionary with two possible value types:
            - For instance classes (is_instance=True): {sem_id: [(instance_id, mesh), ...]}
              List of (instance_id, mesh) tuples
            - For stuff classes (is_instance=False): {sem_id: mesh}
              Single merged mesh with instance_id=0
        bbox: Optional bounding box to visualize (shape: (2, 3) for min/max bounds)
        camera_setting: Optional CameraSetting object for predefined camera configurations
        show_axis: Whether to display the coordinate frame
        colormap: Color mapping scheme. One of 'carla', 'nuscenes', 'kitti360'. Defaults to 'carla'.
        save_path: Optional path to save the visualization as an image
        line_width: Width of wireframe lines
    """
    # Setup visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=camera_setting.image_size[0],
                      height=camera_setting.image_size[1],
                      visible=(save_path is None))

    # Process each semantic class
    for sem_id, mesh_data in pano_mesh.items():
        if mesh_data is None:
            continue

        # Get base semantic color (for stuff classes)
        base_color = np.array(get_color_from_colormap(sem_id, colormap)) / 255.0

        # Check if mesh_data is a list (instance class) or single mesh (stuff class)
        if isinstance(mesh_data, list):
            # Instance-level class: color each instance with deterministic random color
            # Following COCO convention: use global instance_id for color generation
            for instance_id, mesh in mesh_data:
                if mesh is None:
                    continue

                # Create Open3D mesh first
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

                # Create LineSet from mesh edges
                line_set = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)

                # Color all edges with instance-specific color (using INSTANCE_COLOR_OFFSET_LUT, same as vis_pano_occ)
                instance_color = np.clip(
                    base_color + INSTANCE_COLOR_OFFSET_LUT[instance_id % len(INSTANCE_COLOR_OFFSET_LUT)],
                    0, 1)
                num_lines = len(line_set.lines)
                colors = np.tile(instance_color, (num_lines, 1))
                line_set.colors = o3d.utility.Vector3dVector(colors)

                vis.add_geometry(line_set)
        else:
            # Stuff class: single mesh with uniform semantic color
            mesh = mesh_data

            # Create Open3D mesh first
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

            # Create LineSet from mesh edges
            line_set = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)

            # Color all edges with semantic color
            num_lines = len(line_set.lines)
            colors = np.tile(base_color, (num_lines, 1))
            line_set.colors = o3d.utility.Vector3dVector(colors)

            vis.add_geometry(line_set)

    # Add coordinate frame
    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        vis.add_geometry(axis)

    # Add bounding box
    if bbox is not None:
        assert bbox.shape == (2, 3), f"Bbox must have shape (2, 3), got {bbox.shape}"
        min_bound, max_bound = bbox
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
        bbox_lines.paint_uniform_color([1.0, 0.0, 0.0])  # red
        vis.add_geometry(bbox_lines)

    # Configure camera parameters
    ctr = vis.get_view_control()
    camera_params = camera_setting.get_pinhole_camera_parameters()
    ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    # Configure rendering options
    render_option = vis.get_render_option()
    render_option.line_width = line_width

    # Display or save visualization
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis.capture_screen_image(save_path, do_render=True)
    else:
        vis.run()
    vis.destroy_window()


def vis_pano_occ(occ: np.ndarray,
                 voxel_size: float = 0.2,
                 bbox: Optional[np.ndarray] = None,
                 camera_setting: Optional[CameraSetting] = CameraSetting.default_vis_occ(),
                 show_axis: bool = False,
                 colormap: str = 'carla',
                 save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Visualize 3D panoptic occupancy grid using Open3D.
    
    This function creates an interactive 3D visualization of panoptic occupancy where:
    - Instance-level classes (instance_id > 0) are colored with variations of their semantic color
    - Stuff classes (instance_id = 0) use uniform semantic colors
    
    Args:
        occ: Panoptic occupancy grid in Nx4 format (x, y, z, combined_id) where combined_id 
             encodes both semantic_id (high 5 bits) and instance_id (low 11 bits)
        voxel_size: Size of each voxel in meters
        bbox: Optional bounding box to visualize (shape: (2, 3) for min/max bounds)
        camera_setting: Optional CameraSetting object for predefined camera configurations
        show_axis: Whether to display the coordinate frame
        colormap: Color mapping scheme. One of 'carla', 'nuscenes', 'kitti360'. Defaults to 'carla'.
        save_path: Optional path to save visualization image
        
    """
    assert occ.ndim == 3 or (occ.ndim == 2 and occ.shape[1] == 4), \
        "Occupancy grid must be in shape of XxYxZ or Nx4"
    if occ.ndim == 3:
        occ = occ_dense2sparse(occ)

    # Extract coordinates and combined IDs
    points = (occ[:, :3] * voxel_size).astype(np.float32)
    combined_ids = occ[:, 3].astype(np.uint16)

    # Get corresponding colors based on colormap
    semantic_ids, instance_ids = decode_combined_id(combined_ids)

    # Build color LUT for the specified colormap
    if colormap == 'carla':
        colors = color_LUT[semantic_ids]
    elif colormap == 'nuscenes':
        # Build NuScenes color LUT on-the-fly
        max_id = max(nuscenes_id2color.keys())
        nuscenes_color_lut = np.zeros((max_id + 1, 3))
        for idx, color in nuscenes_id2color.items():
            nuscenes_color_lut[idx] = np.array(color) / 255.0
        colors = nuscenes_color_lut[semantic_ids]
    elif colormap == 'kitti360':
        # Build KITTI-360 color LUT on-the-fly
        max_id = max(kitti360_id2color.keys())
        kitti360_color_lut = np.zeros((max_id + 1, 3))
        for idx, color in kitti360_id2color.items():
            kitti360_color_lut[idx] = np.array(color) / 255.0
        colors = kitti360_color_lut[semantic_ids]
    else:
        raise ValueError(f"Unsupported colormap: {colormap}. Choose from 'carla', 'nuscenes', 'kitti360'.")

    colors += INSTANCE_COLOR_OFFSET_LUT[instance_ids % len(INSTANCE_COLOR_OFFSET_LUT)]
    colors = np.clip(colors, 0.0, 1.0)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Convert point cloud to voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    # Visualizer setup
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=camera_setting.image_size[0],
                      height=camera_setting.image_size[1],
                      visible=(save_path is None))
    vis.add_geometry(voxel_grid)

    # Add coordinate frame
    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        vis.add_geometry(axis)

    # Add bounding box
    if bbox is not None:
        assert bbox.shape == (2, 3), f"Bbox must have shape (2, 3), got {bbox.shape}"
        min_bound, max_bound = bbox
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
        bbox_lines.paint_uniform_color([1.0, 0.0, 0.0])  # red
        vis.add_geometry(bbox_lines)

    # Configure camera parameters
    ctr = vis.get_view_control()
    camera_params = camera_setting.get_pinhole_camera_parameters()
    ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    # Display or save visualization
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis.capture_screen_image(save_path, do_render=True)
    else:
        vis.run()
    vis.destroy_window()


def get_instance_color(base_colors, instance_ids) -> np.ndarray:
    colors = base_colors.copy()

    offset_r = np.sin(instance_ids * 0.1) * 0.4
    offset_g = np.sin(instance_ids * 0.3 + 1.0) * 0.4
    offset_b = np.sin(instance_ids * 0.5 + 2.0) * 0.4

    offsets = np.column_stack([offset_r, offset_g, offset_b])

    mask = instance_ids > 0
    colors[mask] = np.clip(colors[mask] + offsets[mask], 0.0, 1.0)
    return colors
