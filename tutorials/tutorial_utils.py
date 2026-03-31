"""Tutorial utility functions for CarlaOcc dataset visualization and analysis.

Provides helpers for:
  1. Image warping (depth-based view synthesis)
  2. 3-D bounding-box projection & drawing
  3. Multi-frame LiDAR aggregation & projection onto images
  4. LiDAR ↔ depth consistency checking
  5. Coordinate transforms  & projection utilities
  6. Depth → point-cloud conversion & Open3D visualization
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import os
from pathlib import Path

import open3d as o3d

from utils.vis_utils import CameraSetting


# ============================================================
# 1.  IMAGE WARP
# ============================================================

def _depth_to_3d(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project a depth image to 3-D camera coordinates.

    Args:
        depth: (H, W) depth map in metres.
        K: (3, 3) camera intrinsic matrix.

    Returns:
        cam_coords: (3, H, W) 3-D coordinates in camera space.
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pix_coords = np.stack([u, v, np.ones_like(u)], axis=0).reshape(3, -1)
    cam_coords = np.linalg.inv(K) @ pix_coords * depth.reshape(1, -1)
    return cam_coords.reshape(3, H, W)


def warp_image(
    src_rgb: np.ndarray,
    src_depth: np.ndarray,
    tgt_depth: np.ndarray,
    pose_tgt2src: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Warp ``src_rgb`` into the viewpoint of the target frame.

    Args:
        src_rgb: (H, W, 3) source RGB image.
        src_depth: (H, W) source depth (metres).
        tgt_depth: (H, W) target depth (metres).
        pose_tgt2src: (3, 4) or (4, 4) target-to-source rigid transform.
        K: (3, 3) shared camera intrinsic matrix.

    Returns:
        warped_rgb, warped_depth, computed_depth, valid_mask
    """
    if pose_tgt2src.shape == (4, 4):
        pose_tgt2src = pose_tgt2src[:3, :]

    H, W = src_rgb.shape[:2]
    p_cam = _depth_to_3d(tgt_depth, K)
    p_cam_h = np.vstack([p_cam.reshape(3, -1), np.ones((1, H * W))])
    p_src = (K @ pose_tgt2src) @ p_cam_h
    z = p_src[2, :]
    uv = p_src[:2, :] / (z + 1e-8)

    map_x = uv[0].reshape(H, W).astype(np.float32)
    map_y = uv[1].reshape(H, W).astype(np.float32)

    warped_rgb = cv2.remap(src_rgb, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
    warped_depth = cv2.remap(src_depth.astype(np.float32), map_x, map_y,
                             interpolation=cv2.INTER_LINEAR)
    computed_depth = z.reshape(H, W)

    valid_mask = (
        (uv[0] >= 0) & (uv[0] < W - 1) &
        (uv[1] >= 0) & (uv[1] < H - 1) &
        (z > 0)
    ).reshape(H, W)

    return warped_rgb, warped_depth, computed_depth, valid_mask


# ============================================================
# 2.  3D BOUNDING BOX DRAWING
# ============================================================

_BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),   # top
    (0, 4), (1, 5), (2, 6), (3, 7),   # pillars
]


def get_bbox_corners_world(
    extent: np.ndarray,
    bbox_transform: np.ndarray,
    object_transform: np.ndarray,
) -> np.ndarray:
    """Compute 8 OBB corners in world coordinates.

    Args:
        extent: (3,) half-extents [l, w, h].
        bbox_transform: (4, 4) bbox → object.
        object_transform: (4, 4) object → world.

    Returns:
        (8, 3) corner positions in world frame.
    """
    l, w, h = extent
    corners = np.array([
        [-l, -w, -h], [l, -w, -h], [l,  w, -h], [-l,  w, -h],
        [-l, -w,  h], [l, -w,  h], [l,  w,  h], [-l,  w,  h],
    ])
    ones = np.ones((8, 1))
    corners_obj = (bbox_transform @ np.hstack([corners, ones]).T).T[:, :3]
    corners_world = (object_transform @ np.hstack([corners_obj, ones]).T).T[:, :3]
    return corners_world


def _world_to_camera(
    points_world: np.ndarray,
    ego_pose: np.ndarray,
    cam_extrinsics: np.ndarray,
) -> np.ndarray:
    """Transform world-frame points to camera frame (X right, Y down, Z fwd)."""
    ones = np.ones((points_world.shape[0], 1))
    pts_ego = (np.linalg.inv(ego_pose) @ np.hstack([points_world, ones]).T).T[:, :3]
    lidar_to_cam = np.linalg.inv(cam_extrinsics)
    pts = (lidar_to_cam @ np.hstack([pts_ego, np.ones((pts_ego.shape[0], 1))]).T).T[:, :3]
    return np.stack([-pts[:, 1], -pts[:, 2], pts[:, 0]], axis=-1)


def draw_3d_bboxes(
    image: np.ndarray,
    bboxes: List[dict],
    ego_pose: np.ndarray,
    cam_extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Project and draw 3-D bounding boxes onto a camera image.

    Args:
        image: (H, W, 3) RGB image (not modified in-place).
        bboxes: List of bbox dicts from ``CarlaOccDataset`` traffic_info.
        ego_pose: (4, 4) ego/LiDAR pose in world.
        cam_extrinsics: (4, 4) camera-to-LiDAR transform.
        intrinsics: (3, 3) camera intrinsic matrix.
        color: RGB draw colour.
        thickness: Line thickness.

    Returns:
        Copy of *image* with boxes drawn.
    """
    out = image.copy()
    for bbox in bboxes:
        if 'bbox' in bbox:
            extent = np.asarray(bbox['bbox']['extent'], dtype=float)
            bbox_tf = np.asarray(bbox['bbox']['transform'], dtype=float)
            obj_tf = np.asarray(bbox['transform'], dtype=float)
        elif 'bbox_extent' in bbox:
            extent = np.asarray(bbox['bbox_extent'], dtype=float)
            bbox_tf = np.eye(4)
            obj_tf = np.asarray(bbox['transform'], dtype=float)
        else:
            continue

        corners_w = get_bbox_corners_world(extent, bbox_tf, obj_tf)
        corners_cam = _world_to_camera(corners_w, ego_pose, cam_extrinsics)
        pts_2d_h = (intrinsics @ corners_cam.T).T
        uv = (pts_2d_h[:, :2] / pts_2d_h[:, 2:3]).astype(int)
        in_front = corners_cam[:, 2] > 0

        for i, j in _BBOX_EDGES:
            if in_front[i] and in_front[j]:
                cv2.line(out, tuple(uv[i]), tuple(uv[j]), color, thickness)

    return out


# ============================================================
# 3.  LIDAR AGGREGATION + PROJECTION ONTO IMAGE
# ============================================================

def aggregate_lidar(
    dataset,
    reference_frame: int,
    frame_range: Tuple[int, int] = (-5, 5),
    frame_step: int = 1,
    filter_dynamic: bool = True,
    dynamic_classes: Optional[List[int]] = None,
) -> np.ndarray:
    """Aggregate multi-frame LiDAR clouds into the reference frame.

    Args:
        dataset: ``CarlaOccDataset`` instance.
        reference_frame: Target frame index.
        frame_range: (start_offset, end_offset) around *reference_frame*.
        frame_step: Step between selected frames.
        filter_dynamic: Remove points whose semantic label is in *dynamic_classes*.
        dynamic_classes: Semantic IDs to remove (default ``[12, 14, 15, 16]``).

    Returns:
        (N, D) aggregated point cloud in reference-frame coordinates.
    """
    if dynamic_classes is None:
        dynamic_classes = [12, 14, 15, 16]

    ref_sample = dataset[reference_frame]
    ref_pose = ref_sample['lidar_pose']

    start = max(0, reference_frame + frame_range[0])
    end = min(len(dataset), reference_frame + frame_range[1] + 1)

    parts: List[np.ndarray] = []
    for fid in range(start, end, frame_step):
        sample = dataset[fid]
        lidar = sample['lidar']
        pts = lidar[:, [0, 1, 2, 5]] if lidar.shape[1] >= 6 else lidar[:, :3]

        if filter_dynamic and dynamic_classes and pts.shape[1] >= 4:
            mask = np.ones(len(pts), dtype=bool)
            for cid in dynamic_classes:
                mask &= (pts[:, -1] != cid)
            pts = pts[mask]

        if fid != reference_frame:
            src_pose = sample['lidar_pose']
            h = np.ones((pts.shape[0], 1))
            pts_w = (src_pose @ np.hstack([pts[:, :3], h]).T).T
            pts_ref = (np.linalg.inv(ref_pose) @ pts_w.T).T
            pts = np.hstack([pts_ref[:, :3], pts[:, 3:]])

        parts.append(pts)

    return np.vstack(parts)


def project_lidar_on_image(
    lidar_points: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    cam_extrinsics: np.ndarray,
    depth_range: Tuple[float, float] = (0.0, 80.0),
    point_radius: int = 3,
    cmap: str = 'turbo',
) -> np.ndarray:
    """Project LiDAR points onto a camera image, coloured by depth.

    Args:
        lidar_points: (N, 3+) in LiDAR/ego coordinates.
        rgb: (H, W, 3) background image.
        intrinsics: (3, 3) camera intrinsic matrix.
        cam_extrinsics: (4, 4) camera-to-LiDAR transform.
        depth_range: (min, max) for colourmap normalisation.
        point_radius: Circle radius in pixels.
        cmap: Matplotlib colourmap name.

    Returns:
        (H, W, 3) RGB overlay image.
    """
    pts_cam = lidar_to_camera(lidar_points[:, :3], cam_extrinsics)
    valid, uv, depths = project_points_to_image(pts_cam, intrinsics, rgb.shape)
    colors = create_depth_colormap(depths, *depth_range, cmap=cmap)
    return overlay_points_on_image(rgb, uv, colors, point_radius)


# ============================================================
# 4.  LIDAR–DEPTH CONSISTENCY
# ============================================================

def depth_to_lidar_pc(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    cam_extrinsics: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    depth_cap: float = 80.0,
) -> np.ndarray:
    """Convert a depth image to a LiDAR-frame point cloud.

    Args:
        depth: (H, W) depth in metres.
        intrinsics: (3, 3) camera intrinsic matrix.
        cam_extrinsics: (4, 4) camera-to-LiDAR transform.
        rgb: Optional (H, W, 3) RGB image for colouring.
        depth_cap: Maximum valid depth.

    Returns:
        (N, 3) or (N, 6) array in LiDAR coordinates.
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    x_cam = (u - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y_cam = (v - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    z_cam = depth

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)
    valid = (z_cam.ravel() > 0) & (z_cam.ravel() <= depth_cap)
    pts_cam = pts_cam[valid]

    # camera → LiDAR
    pts_lidar = camera_to_lidar_pts(pts_cam, cam_extrinsics)

    if rgb is not None:
        colors = rgb.reshape(-1, 3)[valid].astype(float) / 255.0
        return np.hstack([pts_lidar, colors])
    return pts_lidar


def check_lidar_depth_consistency(
    lidar_points: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    cam_extrinsics: np.ndarray,
    depth_range: Tuple[float, float] = (0.0, 80.0),
    tolerance: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compare LiDAR depth against rendered depth for each camera pixel.

    Returns:
        error_map: (H, W) absolute depth error (NaN where no LiDAR hit).
        overlay: (H, W, 3) error heatmap.
        consistency_ratio: fraction of points with |error| < *tolerance*.
    """
    import matplotlib.pyplot as plt

    H, W = depth.shape
    pts_cam = lidar_to_camera(lidar_points[:, :3], cam_extrinsics)
    _, uv, l_depth = project_points_to_image(pts_cam, intrinsics, (H, W))

    # keep only valid projections (project_points_to_image already filters)
    error_map = np.full((H, W), np.nan, dtype=np.float32)
    us, vs = uv[:, 0], uv[:, 1]
    rd = depth[vs, us]
    valid_hit = rd > 0
    error_map[vs[valid_hit], us[valid_hit]] = np.abs(l_depth[valid_hit] - rd[valid_hit])

    valid_err = error_map[~np.isnan(error_map)]
    consistency = float(np.mean(valid_err < tolerance)) if len(valid_err) else 0.0

    vmax = max(float(np.nanpercentile(error_map, 95)), 1e-6) if len(valid_err) else 1.0
    norm_err = np.nan_to_num(error_map / vmax, nan=0.0)
    heatmap = (plt.get_cmap('hot')(norm_err)[:, :, :3] * 255).astype(np.uint8)

    grey = np.full((H, W, 3), 60, dtype=np.uint8)
    has_data = ~np.isnan(error_map)
    overlay = grey.copy()
    overlay[has_data] = heatmap[has_data]

    return error_map, overlay, consistency


# ============================================================
# 5.  COORDINATE TRANSFORMS & PROJECTION UTILITIES
# ============================================================

def lidar_to_camera(
    pts_lidar: np.ndarray,
    cam_extrinsics: np.ndarray,
) -> np.ndarray:
    """Transform LiDAR-frame points to camera coordinates.

    Coordinate convention::
        LiDAR : X forward, Y left,  Z up
        Camera: X right,   Y down,  Z forward

    Args:
        pts_lidar: (N, 3+) points in LiDAR frame.
        cam_extrinsics: (4, 4) camera-to-LiDAR transform (inverted internally).

    Returns:
        (N, 3) points in camera coordinates.
    """
    pts = pts_lidar[:, :3]
    lidar_to_cam = np.linalg.inv(cam_extrinsics)
    h = np.ones((pts.shape[0], 1))
    pts_cam_axes = (lidar_to_cam @ np.hstack([pts, h]).T).T[:, :3]
    return np.stack([-pts_cam_axes[:, 1], -pts_cam_axes[:, 2], pts_cam_axes[:, 0]], axis=-1)


def camera_to_lidar_pts(
    camera_points: np.ndarray,
    cam_to_lidar_transform: np.ndarray,
) -> np.ndarray:
    """Transform camera-frame points back to LiDAR coordinates.

    Inverse of :func:`lidar_to_camera`.  Preserves extra attribute columns.

    Args:
        camera_points: (N, 3+) with first 3 cols as XYZ in camera frame.
        cam_to_lidar_transform: (4, 4) camera-to-LiDAR transform.

    Returns:
        (N, M) array — first 3 cols are LiDAR XYZ, rest copied from input.
    """
    xyz = camera_points[:, :3]
    h = np.ones((xyz.shape[0], 1))
    lidar_xyz = (cam_to_lidar_transform @ np.hstack([xyz, h]).T).T[:, :3]
    if camera_points.shape[1] > 3:
        return np.hstack([lidar_xyz, camera_points[:, 3:]])
    return lidar_xyz


def project_points_to_image(
    pts_cam: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple,
    min_depth: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame 3-D points onto the image plane.

    Args:
        pts_cam: (N, 3) in camera coords (X right, Y down, Z fwd).
        intrinsics: (3, 3) camera intrinsic matrix.
        image_shape: (H, W, ...).
        min_depth: Keep only points with Z > *min_depth*.

    Returns:
        valid_mask (N,), uv (M, 2) int, depths (M,).
    """
    H, W = image_shape[:2]
    in_front = pts_cam[:, 2] > min_depth
    pts_f = pts_cam[in_front]

    pts_2d_h = (intrinsics @ pts_f.T).T
    uv_all = (pts_2d_h[:, :2] / pts_2d_h[:, 2:3]).astype(int)
    depths_all = pts_f[:, 2]

    in_img = (uv_all[:, 0] >= 0) & (uv_all[:, 0] < W) & \
             (uv_all[:, 1] >= 0) & (uv_all[:, 1] < H)

    valid_mask = np.zeros(len(pts_cam), dtype=bool)
    valid_mask[np.where(in_front)[0][in_img]] = True

    return valid_mask, uv_all[in_img], depths_all[in_img]


def create_depth_colormap(
    depths: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 80.0,
    cmap: str = 'turbo',
) -> np.ndarray:
    """Map depth values to (N, 3) uint8 RGB colours via a matplotlib colourmap."""
    import matplotlib.pyplot as plt
    norm = np.clip((depths - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgba = plt.get_cmap(cmap)(norm)
    return (rgba[:, :3] * 255).astype(np.uint8)


def overlay_points_on_image(
    image: np.ndarray,
    uv: np.ndarray,
    colors: np.ndarray,
    point_size: int = 2,
) -> np.ndarray:
    """Draw coloured circles on an image at given pixel locations.

    Args:
        image: (H, W, 3) RGB background.
        uv: (N, 2) int pixel coordinates.
        colors: (N, 3) uint8 RGB colours.
        point_size: Circle radius in pixels.

    Returns:
        (H, W, 3) overlay image.
    """
    overlay = image.copy()
    if len(uv) == 0:
        return overlay
    h, w = overlay.shape[:2]
    for dy in range(-point_size, point_size + 1):
        for dx in range(-point_size, point_size + 1):
            if dx * dx + dy * dy <= point_size * point_size:
                ys = np.clip(uv[:, 1] + dy, 0, h - 1)
                xs = np.clip(uv[:, 0] + dx, 0, w - 1)
                overlay[ys, xs] = colors
    return overlay


# ============================================================
# 6.  DEPTH → POINT CLOUD  &  OPEN3D VISUALIZATION
# ============================================================

def depth2pc(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    semantics: Optional[np.ndarray] = None,
    instance_mask: Optional[np.ndarray] = None,
    down_sample_ratio: Optional[int] = None,
    depth_cap: float = 80.0,
) -> np.ndarray:
    """Project a depth image to a 3-D point cloud with optional attributes.

    Returns:
        (N, 10) array: [X, Y, Z, R, G, B, sem_R, sem_G, sem_B, instance_id].
    """
    assert depth.ndim == 2
    assert intrinsics.shape == (3, 3)
    h, w = depth.shape

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * depth / intrinsics[1, 1]

    # Convert to LiDAR coordinates (z, –x, –y)
    points_3d = np.stack((depth, -x, -y), axis=-1)

    if down_sample_ratio is not None and down_sample_ratio >= 1:
        s = down_sample_ratio
        points_3d = points_3d[::s, ::s]
        rgb = rgb[::s, ::s] if rgb is not None else None
        semantics = semantics[::s, ::s] if semantics is not None else None
        instance_mask = instance_mask[::s, ::s] if instance_mask is not None else None

    valid = ~np.isnan(points_3d).any(axis=-1) & \
            (points_3d[..., 0] > 0) & (points_3d[..., 0] < depth_cap)
    vp = points_3d[valid]

    def _get(arr, default_cols):
        return arr[valid] if arr is not None else np.zeros((vp.shape[0], default_cols))

    return np.hstack([
        vp,
        _get(rgb, 3),
        _get(semantics, 3),
        _get(instance_mask, 1) if instance_mask is None
        else instance_mask[valid, None] if instance_mask.ndim == 2
        else instance_mask[valid].reshape(-1, 1),
    ])


def vis_pc(
    pc: np.ndarray,
    color_mode: str = 'semantics',
    camera_setting: Optional[CameraSetting] = None,
    show_axis: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Visualize a point cloud with Open3D.

    Args:
        pc: (N, M) array with M ≥ 3.  Column order: XYZ, RGB, semantics, instance.
        color_mode: 'rgb', 'semantics', or anything else for grey.
        camera_setting: Open3D camera settings (defaults to ``CameraSetting.default_setting()``).
        show_axis: Show coordinate frame.
        save_path: If set, save screenshot instead of interactive display.
    """
    if camera_setting is None:
        camera_setting = CameraSetting.default_setting()

    assert pc.ndim == 2 and pc.shape[1] >= 3

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])

    if color_mode == 'rgb' and pc.shape[1] >= 6:
        colors = pc[:, 3:6] / 255.0
    elif color_mode == 'semantics' and pc.shape[1] >= 9:
        colors = pc[:, 6:9] / 255.0
    else:
        colors = np.full_like(pc[:, :3], 0.5)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=camera_setting.image_size[0],
                      height=camera_setting.image_size[1],
                      visible=(save_path is None))
    vis.add_geometry(pcd)

    if show_axis:
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(
        camera_setting.get_pinhole_camera_parameters(), allow_arbitrary=True)

    if save_path:
        parent = os.path.dirname(str(save_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        vis.capture_screen_image(str(save_path), do_render=True)
    else:
        vis.run()
    vis.destroy_window()
