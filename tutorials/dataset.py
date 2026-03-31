from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import yaml

from utils.load_utils import (
    get_image_path,
    load_rgb,
    load_depth,
    load_semantics,
    load_lidar,
    load_semantic_lidar,
    load_traffic_info,
    load_pano_occ,
    load_calibration,
)


class CarlaOccDataset:
    """CarlaOcc dataset loader.

    Reads a YAML config that specifies data_root, split, camera layout,
    and which modalities to load.  Each ``dataset[i]`` returns a dict
    with all requested data for one frame.
    """

    def __init__(self, config_path: Union[Path, str]):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_root = Path(self.config["data_root"])
        self.split_dir = self.data_root / 'splits'
        self.dataset_type = self.config["dataset_type"]
        self.loading_type = self.config["loading_type"]

        # --- parse split file ---
        self.seq_frames: Dict[str, List[int]] = defaultdict(list)
        with open(self.split_dir / f'{self.dataset_type}.txt', 'r') as f:
            for line in f:
                seq_name, frame_id = self._parse_frame_line(line)
                self.seq_frames[seq_name].append(int(frame_id))
        self.seq_frames = {s: sorted(ids) for s, ids in self.seq_frames.items()}

        # --- cameras ---
        self.cameras = self._get_camera_list()

        # --- calibration ---
        self.calib_data = load_calibration(
            self.data_root / "calib" / "calib.yaml", self.cameras)

        # --- LiDAR poses ---
        self.seq_names = list(self.seq_frames.keys())
        self.lidar_poses: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)
        for seq in self.seq_names:
            pose_data = np.loadtxt(
                self.data_root / seq / "poses" / "lidar.txt", dtype=np.float32)
            self.lidar_poses[seq] = {
                int(arr[0]): arr[1:].reshape(4, 4) for arr in pose_data
            }

        # --- data infos ---
        self.data_infos = self._build_data_infos()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_frame_line(line: str) -> tuple:
        """Parse 'town_num seq_num frame_id' → (seq_name, frame_id)."""
        town_num, seq_num, frame_id_str = line.split()
        frame_id = int(frame_id_str)
        prefix = 'Town10HD_Opt' if town_num == '10' else f'Town{town_num}_Opt'
        return f'{prefix}_Seq{seq_num}', frame_id

    def _get_camera_list(self) -> List[str]:
        """Return camera list based on ``loading_type``."""
        camera_configs = {
            "mono": ["cam_00"],
            "stereo": ["cam_00", "cam_01"],
            "kitti360-like": ["cam_00", "cam_01", "cam_02", "cam_03"],
            "surrounding": [f"cam_{i:02d}" for i in range(6)],
        }
        return camera_configs[self.loading_type]

    def _build_data_infos(self) -> List[Dict[str, Any]]:
        """Build per-frame metadata list."""
        infos: List[Dict[str, Any]] = []
        for seq, frame_ids in self.seq_frames.items():
            poses = self.lidar_poses[seq]
            for fid in sorted(set(frame_ids)):
                infos.append({
                    'seq_name': seq,
                    'frame_id': fid,
                    'lidar_pose': poses[fid],
                    'timestamp': fid * 0.1,
                })
        return infos

    # ------------------------------------------------------------------ #
    #  Traffic info helpers
    # ------------------------------------------------------------------ #

    def _transform_bbox_to_sensor(
        self, object_data: Dict, ego_pose: np.ndarray,
        sensor_transform: np.ndarray, is_pedestrian: bool = False,
    ) -> Dict[str, Any]:
        """Transform bbox: object → world → ego → sensor."""
        if is_pedestrian:
            extent = np.array(object_data["bbox_extent"])
            bbox_transform = np.eye(4)
        else:
            bbox_info = object_data["bbox"]
            extent = np.array(bbox_info["extent"])
            bbox_transform = np.array(bbox_info["transform"])

        obj_world = np.array(object_data["transform"])
        bbox_world = obj_world @ bbox_transform
        bbox_ego = np.linalg.inv(ego_pose) @ bbox_world
        bbox_sensor = np.linalg.inv(sensor_transform) @ bbox_ego

        return {
            "extent": extent.tolist(),
            "transform": bbox_sensor.tolist(),
            "object_id": object_data["id"],
            "object_type": object_data["type"],
        }

    def _process_traffic_bboxes(
        self, seq_name: str, traffic_info: Dict, frame_id: int,
    ) -> List[Dict[str, Any]]:
        """Transform vehicle / pedestrian bboxes into each camera frame."""
        results: List[Dict[str, Any]] = []
        ego_pose = self.lidar_poses[seq_name][frame_id]

        for category, is_ped in [("vehicles", False), ("pedestrians", True)]:
            key_check = "bbox_extent" if is_ped else "bbox"
            for obj in traffic_info.get(category, []):
                if key_check not in obj:
                    continue
                cam_bboxes = {}
                for cam in self.cameras:
                    ext = self.calib_data['extrinsics'][cam]
                    bbox = self._transform_bbox_to_sensor(obj, ego_pose, ext, is_ped)
                    if bbox is not None:
                        cam_bboxes[cam] = bbox
                if cam_bboxes:
                    entry: Dict[str, Any] = {
                        "id": obj["id"],
                        "type": obj["type"],
                        "distance_to_ego": obj.get("distance_to_ego", 0.0),
                        "bboxes": cam_bboxes,
                    }
                    if not is_ped:
                        entry["velocity"] = obj.get("velocity", [0.0, 0.0, 0.0])
                    results.append(entry)
        return results

    def _load_traffic_info(self, seq_name: str, frame_id: int):
        """Load + transform traffic info for one frame."""
        path = self.data_root / seq_name / "traffic_info" / f"{frame_id:04d}.yaml"
        info = load_traffic_info(path)
        return self._process_traffic_bboxes(seq_name, info, frame_id)

    # ------------------------------------------------------------------ #
    #  Sweep collection (multi-frame)
    # ------------------------------------------------------------------ #

    def collect_sweeps(self, index: int, past_num: int = 7, future_num: int = 0):
        """Collect frame metadata for neighbouring frames in the same sequence.

        Returns:
            past_sweeps: list of up to *past_num* ``data_infos`` dicts, newest-first.
            future_sweeps: list of up to *future_num* ``data_infos`` dicts, oldest-first.
        """
        seq_name = self.data_infos[index]['seq_name']

        past_sweeps: List[Dict[str, Any]] = []
        for offset in range(1, past_num + 1):
            idx = index - offset
            if idx < 0 or self.data_infos[idx]['seq_name'] != seq_name:
                break
            past_sweeps.append(self.data_infos[idx])

        future_sweeps: List[Dict[str, Any]] = []
        for offset in range(1, future_num + 1):
            idx = index + offset
            if idx >= len(self.data_infos) or self.data_infos[idx]['seq_name'] != seq_name:
                break
            future_sweeps.append(self.data_infos[idx])

        return past_sweeps, future_sweeps

    # ------------------------------------------------------------------ #
    #  Public interface
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]
        seq = info['seq_name']
        fid = info['frame_id']
        cfg = self.config

        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        # ---- conditional loading via shared load_utils functions ----
        def _load_cameras(loader):
            return {cam: loader(self.data_root, seq, fid, cam) for cam in self.cameras}

        rgb = _load_cameras(load_rgb) if cfg.get("load_rgb") else None
        depth = _load_cameras(load_depth) if cfg.get("load_depth") else None
        semantics = _load_cameras(load_semantics) if cfg.get("load_semantic") else None

        lidar = None
        if cfg.get("load_lidar"):
            lidar = load_lidar(self.data_root / seq / "lidar" / f"{fid:04d}.ply")

        semantic_lidar = None
        if cfg.get("load_semantic_lidar"):
            semantic_lidar = load_semantic_lidar(
                self.data_root / seq / "semantic_lidar" / f"{fid:04d}.ply")

        pano_occ = load_pano_occ(self.data_root, seq, fid, cfg) \
            if cfg.get("load_pano_occ") else None

        traffic = self._load_traffic_info(seq, fid) \
            if cfg.get("load_traffic_info") else None

        return {
            'seq_name': seq,
            'frame_id': fid,
            'rgb': rgb,
            'depth': depth,
            'semantics': semantics,
            'lidar': lidar,
            'semantic_lidar': semantic_lidar,
            'pano_occ': pano_occ,
            'traffic_info': traffic,
            'lidar_pose': info['lidar_pose'],
            'timestamp': info['timestamp'],
            'intrinsics': self.calib_data['intrinsics'],
            'extrinsics': self.calib_data['extrinsics'],
            'image_size': (self.calib_data['height'], self.calib_data['width']),
            'sweeps': sweeps_prev,
        }


if __name__ == '__main__':
    dataset = CarlaOccDataset("tutorials/config/dataset.yaml")
    sample = dataset[0]
    print(f"Sample data keys: {list(sample.keys())}")
    print(f"Image size: {sample['image_size']}")
