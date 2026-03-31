from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import yaml
from plyfile import PlyData

from utils.data_converter import decode_combined_id
from utils.occ_ops import occ_sparse2dense


def get_image_path(data_root: Path, seq_name: str, frame_id: int,
                   camera: str, data_type: str) -> Optional[Path]:
    """Build file path for a specific camera image.

    Args:
        data_root: Root directory of the dataset.
        seq_name: Sequence name (e.g. 'Town01_Opt_Seq07').
        frame_id: Frame index.
        camera: Camera name (e.g. 'cam_00').
        data_type: One of 'rgb', 'depth', 'semantic'.

    Returns:
        Path object if the file exists, otherwise ``None``.
    """
    # Map shorthand data types to actual directory names
    dir_name = {"rgb": "rgb", "depth": "depth", "semantics": "semantics"}.get(data_type, data_type)
    camera_num = camera.split('_')[-1]
    file_path = data_root / seq_name / dir_name / f"image_{camera_num}" / f"{frame_id:04d}.png"
    return file_path if file_path.exists() else None


def load_rgb(data_root: Path, seq_name: str, frame_id: int,
             camera: str) -> Optional[np.ndarray]:
    """Load an RGB image and convert from BGR to RGB."""
    file_path = get_image_path(data_root, seq_name, frame_id, camera, "rgb")
    if file_path is None:
        return None
    img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(data_root: Path, seq_name: str, frame_id: int,
               camera: str) -> Optional[np.ndarray]:
    """Load a 16-bit depth image and scale to metres (0–80 m)."""
    file_path = get_image_path(data_root, seq_name, frame_id, camera, "depth")
    if file_path is None:
        return None
    depth = cv2.imread(str(file_path), -1)
    return depth.astype(np.float32) / 65535 * 80.0


def load_semantics(data_root: Path, seq_name: str, frame_id: int,
                   camera: str) -> Optional[np.ndarray]:
    """Load a semantic segmentation image (BGR → RGB)."""
    file_path = get_image_path(data_root, seq_name, frame_id, camera, "semantics")
    if file_path is None:
        return None
    seg = cv2.imread(str(file_path), -1)
    return cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)


def load_lidar(lidar_path: Path) -> Optional[np.ndarray]:
    """Load a LiDAR point cloud from a PLY file.

    Returns:
        (N, 4) array — [x, y, z, intensity].
    """
    if not Path(lidar_path).exists():
        return None
    ply_data = PlyData.read(str(lidar_path))
    vertex = ply_data['vertex']
    return np.column_stack([vertex['x'], vertex['y'], vertex['z'], vertex['I']])


def load_semantic_lidar(semantic_lidar_path: Path) -> Optional[np.ndarray]:
    """Load a semantic LiDAR point cloud from a PLY file.

    Returns:
        (N, 6) array — [x, y, z, CosAngle, ObjIdx, ObjTag].
    """
    if not Path(semantic_lidar_path).exists():
        return None
    ply_data = PlyData.read(str(semantic_lidar_path))
    vertex = ply_data['vertex']
    return np.column_stack([
        vertex['x'], vertex['y'], vertex['z'],
        vertex['CosAngle'], vertex['ObjIdx'], vertex['ObjTag'],
    ])


def load_traffic_info(traffic_info_path: Path) -> Optional[Dict[str, Any]]:
    """Load traffic information YAML for a single frame."""
    with open(traffic_info_path, 'r') as f:
        return yaml.safe_load(f)


def load_pano_occ(data_root: Path, seq_name: str, frame_id: int,
                  config: Dict) -> Dict[str, Any]:
    """Load and decode panoptic occupancy data for a specific frame.

    Returns:
        Dict with keys: ``semantics``, ``instances``, ``voxel_size``,
        ``voxel_origin``, ``volume_size``.
    """
    occ_file = (data_root / seq_name / "occupancy"
                / config['occ_config']['type'] / f"{frame_id:04d}.npz")
    occ_data = np.load(occ_file)
    voxel_size = occ_data['voxel_size']
    voxel_origin = occ_data['voxel_origin']
    volume_size = occ_data['volume_size']

    dense_occupancy = occ_sparse2dense(occ_data['occupancy'],
                                       empty_value=0, volume_size=volume_size)
    semantics, instances = decode_combined_id(dense_occupancy)

    return {
        'semantics': semantics,
        'instances': instances,
        'voxel_size': voxel_size,
        'voxel_origin': voxel_origin,
        'volume_size': volume_size,
    }


def load_calibration(calib_path, cameras: List[str]) -> Dict[str, Any]:
    """Load calibration data (intrinsics + per-camera extrinsics) from YAML."""
    with open(calib_path, 'r') as f:
        calib = yaml.safe_load(f)

    cam = calib['cam_settings']
    return {
        'width': cam['width'],
        'height': cam['height'],
        'fov': cam['fov'],
        'intrinsics': np.array(cam['intrinsics'], dtype=np.float32),
        'extrinsics': {
            c: np.array(calib['sensors'][c]['transform'], dtype=np.float32)
            for c in cameras
        },
    }
