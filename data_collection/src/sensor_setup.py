import logging
import time
from typing import Dict

import carla

from utils.transforms import dict_to_carla_transform

logger = logging.getLogger(__name__)


class SynchronousSensors:
    def __init__(self, world, vehicle, sensor_config):
        self.world = world
        self.vehicle = vehicle
        self.sensor_config = sensor_config
        self.sensors: Dict[str, carla.Actor] = {}

        # Initialize sensors
        self._setup_sensors()

    def _setup_sensors(self):
        for k, v in self.sensor_config.items():
            if 'camera' in k:
                self._setup_camera(k, v)
            elif 'lidar' in k:
                self._setup_lidar(k, v)
        logger.info("sensors setup done...")

    def _setup_camera(self, camera_name, camera_config):
        """Setup a camera sensor."""
        fov = camera_config['fov']
        image_size_x = camera_config['image_size_x']
        image_size_y = camera_config['image_size_y']
        sensor_types = camera_config['type']
        transform = dict_to_carla_transform(camera_config['transform'])

        for t in sensor_types:
            camera_bp = self.world.get_blueprint_library().find(f'sensor.camera.{t}')
            camera_bp.set_attribute('image_size_x', str(image_size_x))
            camera_bp.set_attribute('image_size_y', str(image_size_y))
            camera_bp.set_attribute('fov', str(fov))
            if t == 'rgb':
                camera_bp.set_attribute('exposure_mode', 'manual')

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
            self.sensors[f'{camera_name}_{t}'] = camera

    def _setup_lidar(self, lidar_name, lidar_config):
        """Setup a LiDAR sensor."""
        channels = lidar_config['channels']
        points_per_second = lidar_config['points_per_second']
        rotation_frequency = lidar_config['rotation_frequency']
        sensor_range = lidar_config['range']
        upper_fov = lidar_config['upper_fov']
        lower_fov = lidar_config['lower_fov']
        sensor_types = lidar_config['type']
        transform = dict_to_carla_transform(lidar_config['transform'])

        for t in sensor_types:
            lidar_bp = self.world.get_blueprint_library().find(f'sensor.lidar.{t}')
            lidar_bp.set_attribute('channels', str(channels))
            lidar_bp.set_attribute('points_per_second', str(points_per_second))
            lidar_bp.set_attribute('rotation_frequency', str(rotation_frequency))
            lidar_bp.set_attribute('range', str(sensor_range))
            lidar_bp.set_attribute('upper_fov', str(upper_fov))
            lidar_bp.set_attribute('lower_fov', str(lower_fov))

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=self.vehicle)
            self.sensors[f'{lidar_name}_{t}'] = lidar

    def destroy(self):
        """Destroy all sensors."""
        for sensor in self.sensors.values():
            if sensor is not None and sensor.is_alive:
                sensor.destroy()
        self.sensors = {}
        for _ in range(2):
            self.world.tick()
            time.sleep(0.5)
        logger.info("Sensors destroyed.")
