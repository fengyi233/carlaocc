import shutil
from pathlib import Path

import carla
import cv2
import numpy as np
import yaml

from utils.save_utils import formatter
from utils.transforms import carla_transform_to_mat, mat_left_to_right, vector_left_to_right, carla_to_np


class DataSaver:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.save_dir = None

    def make_dirs(self, sub_dir):
        """Create necessary directories for data storage."""
        self.save_dir = self.dataset_dir / sub_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create main directories
        image_dirs = ['image_00', 'image_01', 'image_02', 'image_03', 'image_04', 'image_05']
        dirs = {
            'rgb': image_dirs + ['image_bev'],
            'semantics_carla': image_dirs,
            'depth_carla': image_dirs,
            'lidar': [],
            'semantic_lidar': [],
            'traffic_info': [],
            'poses': [],
        }

        # Clear poses directory (pose files are opened in append mode)
        poses_dir = self.save_dir / 'poses'
        if poses_dir.exists():
            shutil.rmtree(poses_dir)

        # Create all directories
        for main_dir, sub_dirs in dirs.items():
            if sub_dirs:
                for sub_dir in sub_dirs:
                    (self.save_dir / main_dir / sub_dir).mkdir(parents=True, exist_ok=True)
            else:
                (self.save_dir / main_dir).mkdir(parents=True, exist_ok=True)

    def save_data(self, frame_id, sensor_data_dict):
        """
        Save sensor data to specified directory.
        
        Args:
            frame_id: Current frame number
            sensor_data_dict: Dictionary of {sensor_name: data}
        """
        frame_str = f"{frame_id:04d}"

        for sensor_name, data in sensor_data_dict.items():
            # save traffic info
            if sensor_name == 'traffic':
                self.save_traffic_info(frame_id, data)

            # save poses
            elif sensor_name == 'poses':
                self.save_poses(frame_id, data)

            # save camera images
            elif sensor_name.startswith('camera_'):
                cam_id, cam_type = sensor_name.split('_')[1], sensor_name.split('_', 2)[-1]

                save_path = self.save_dir / cam_type / f'image_{cam_id}' / f'{frame_str}.png'

                if cam_type == 'rgb':
                    data.convert(carla.ColorConverter.Raw)
                    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (data.height, data.width, 4))[:, :, :3]
                    cv2.imwrite(str(save_path), cv2.convertScaleAbs(array, alpha=1.2, beta=0))
                elif cam_type == 'semantic_segmentation':
                    data.convert(carla.ColorConverter.CityScapesPalette)
                    array = np.frombuffer(data.raw_data, dtype=np.uint8)
                    array = np.reshape(array, (data.height, data.width, 4))[:, :, :3]
                    save_path = self.save_dir / 'semantics_carla' / f'image_{cam_id}' / f'{frame_str}.png'
                    cv2.imwrite(str(save_path), array)
                elif cam_type == 'depth':
                    data.convert(carla.ColorConverter.Raw)
                    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (data.height, data.width, 4))[:, :, :3].astype(np.float32)
                    B, G, R = array[..., 0], array[..., 1], array[..., 2]
                    normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)  # [0,1]
                    depth = 1000 * normalized  # depth in meters
                    depth = np.clip(depth, 0, 80.0) / 80.0
                    depth = (depth * 65535).astype(np.uint16)

                    save_path = self.save_dir / 'depth_carla' / f'image_{cam_id}' / f'{frame_str}.png'
                    cv2.imwrite(str(save_path), depth)

                elif cam_type == 'optical_flow':
                    image = data.get_color_coded_flow()
                    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                    array = array.reshape(image.height, image.width, 4)[..., [2, 1, 0]]  # BGRA -> RGB
                    cv2.imwrite(save_path, array)
                else:
                    raise ValueError(f"Unknown camera type: {cam_type}")

            # save ply files
            elif sensor_name.startswith('lidar_'):
                self.save_lidar(data, frame_id, sensor_name)

            else:
                raise ValueError(f"Unknown sensor type: {sensor_name}")

    def save_traffic_info(self, frame_id, data):
        """
        Save traffic info in right-handed coordinate system.
        """
        data_rh = data.copy()
        data_rh['ego_vehicles']['velocity'] = vector_left_to_right(
            data['ego_vehicles']['velocity'])
        data_rh['ego_vehicles']['acceleration'] = vector_left_to_right(
            data['ego_vehicles']['acceleration'])
        data_rh['ego_vehicles']['angular_velocity'] = vector_left_to_right(
            data['ego_vehicles']['angular_velocity'])

        for vehicle in data_rh['vehicles']:
            vehicle['transform'] = mat_left_to_right(np.array(vehicle['transform']))
            vehicle['velocity'] = vector_left_to_right(vehicle['velocity'])
            vehicle['bbox'] = {
                'extent': carla_to_np(vehicle['bbox'].extent),
                'transform': carla_transform_to_mat(
                    carla.Transform(vehicle['bbox'].location, vehicle['bbox'].rotation)
                )
            }
        for pedestrian in data_rh['pedestrians']:
            pedestrian['transform'] = mat_left_to_right(np.array(pedestrian['transform']))
            pedestrian['bbox_extent'] = carla_to_np(pedestrian['bbox_extent'])
            pedestrian['bones'] = {
                name: mat_left_to_right(np.array(bone)) for name, bone in pedestrian['bones'].items()
            }

        save_path = self.save_dir / 'traffic_info' / f'{frame_id:04d}.yaml'
        with open(save_path, 'w') as f:
            yaml.dump(formatter(data_rh), f, sort_keys=False)

    def save_poses(self, frame_id, data):
        for sensor, transform in data.items():
            save_path = self.save_dir / 'poses' / f'{sensor}.txt'
            mat = carla_transform_to_mat(transform)

            # Flatten to 1x16 and save as txt
            transform_flat = mat.flatten()
            transform_str = ' '.join([f'{x:.6f}' for x in transform_flat])
            transform_str = f'{frame_id:04d} {transform_str}\n'
            with open(save_path, 'a') as f:
                f.write(transform_str)

    def save_lidar(self, data, frame_id, sensor_name):
        # Save LiDAR point cloud
        if sensor_name == 'lidar_ray_cast':
            save_path = self.save_dir / 'lidar' / f'{frame_id:04d}.ply'
            # data.save_to_disk(str(save_path))
            points = np.frombuffer(data.raw_data, dtype=np.float32).copy()
            points = np.reshape(points, (-1, 4))  # [x, y, z, intensity]
            points[:, 1] *= -1

            header = (f"ply\nformat ascii 1.0\nelement vertex {points.shape[0]}\n"
                      f"property float32 x\nproperty float32 y\nproperty float32 z\nproperty float32 I\nend_header")

            with open(save_path, 'w') as f:
                f.write(header + '\n')
                np.savetxt(f, points, fmt='%.4f %.4f %.4f %.4f')
        elif sensor_name == 'lidar_ray_cast_semantic':
            save_path = self.save_dir / 'semantic_lidar' / f'{frame_id:04d}.ply'
            raw = np.frombuffer(data.raw_data, dtype=np.float32)
            raw = raw.reshape(-1, 6)  # [x, y, z, CosAngle, ObjIdx, ObjTag]
            points_float32 = raw[:, :4].copy()
            points_float32[:, 1] *= -1
            # float32 to int32
            points_int32 = raw[:, 4:].view(np.int32).copy()
            points = np.hstack((points_float32, points_int32))

            header = (f"ply\nformat ascii 1.0\nelement vertex {points.shape[0]}\n"
                      f"property float x\nproperty float y\nproperty float z\n"
                      f"property float CosAngle\nproperty int ObjIdx\nproperty int ObjTag\nend_header")
            with open(save_path, 'w') as f:
                f.write(header + '\n')
                np.savetxt(f, points, fmt='%.4f %.4f %.4f %.4f %d %d')
        else:
            raise NotImplementedError(f'Unknown sensor type: {sensor_name}')
