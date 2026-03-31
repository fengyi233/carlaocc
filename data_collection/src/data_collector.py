import logging
import threading
import time
from collections import defaultdict

import carla
from tqdm import tqdm

from data_collection.src.data_saver import DataSaver
from data_collection.src.traffic_setup import TrafficLightController

logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self, world, sensors, traffic_generator, config):
        self.world = world
        self.sensors = sensors  # {"cam_xx_type": Actor}
        self.traffic_generator = traffic_generator
        self.config = config

        self.town_name = world.get_map().name.split('/')[-1]
        self.data_saver = DataSaver(self.config['output_dir'])

        self.frame_data = {}  # {frame_id: {sensor_name: data}}
        self.current_frame = 0
        self.expected_sensors = set(sensors.keys())
        self.data_lock = threading.Lock()

        # Timeout settings
        self.sensor_timeout = 5.0  # Maximum time to wait for sensor data (seconds)
        self.wait_interval = 0.01  # Sleep interval when waiting for sensors
        self.max_wait_iterations = int(self.sensor_timeout / self.wait_interval)

        self.traffic_light_controller = TrafficLightController(world,
                                                               self.traffic_generator.ego_vehicle,
                                                               self.config['traffic_light_config'])

    def sensor_callback(self, data, sensor_type):
        frame_id = getattr(data, 'frame', self.current_frame)
        with self.data_lock:
            if frame_id not in self.frame_data:
                self.frame_data[frame_id] = {}
            self.frame_data[frame_id][sensor_type] = data

    def register_sensor_listeners(self, sensors=None):
        """Register listen callbacks for all sensors and put data into DataCollector queues."""
        if sensors is None:
            sensors = self.sensors
        for name, sensor in sensors.items():
            sensor.listen(lambda data, n=name: self.sensor_callback(data, n))

    def wait_for_sensor_data(self, frame_id):
        """Wait for all sensor data to be ready"""
        start_time = time.time()
        wait_count = 0

        while wait_count < self.max_wait_iterations:
            with self.data_lock:
                # Check if current frame has data
                if frame_id in self.frame_data:
                    received_sensors = set(self.frame_data[frame_id].keys())
                    missing_sensors = self.expected_sensors - received_sensors

                    if not missing_sensors:
                        # All sensor data has been received
                        return True, set()

                    # Check timeout
                    elapsed_time = time.time() - start_time
                    if elapsed_time > self.sensor_timeout:
                        logger.warning(f"Frame {frame_id}: Timeout waiting for sensors: {missing_sensors}")
                        return False, missing_sensors
                else:
                    # Current frame has no data yet
                    elapsed_time = time.time() - start_time
                    if elapsed_time > self.sensor_timeout:
                        logger.warning(f"Frame {frame_id}: Timeout - no sensor data received")
                        return False, self.expected_sensors

            # Brief wait
            time.sleep(self.wait_interval)
            wait_count += 1
        # Reached maximum wait iterations
        with self.data_lock:
            if frame_id in self.frame_data:
                received_sensors = set(self.frame_data[frame_id].keys())
                missing_sensors = self.expected_sensors - received_sensors
            else:
                missing_sensors = self.expected_sensors
        logger.warning(f"Frame {frame_id}: Max wait iterations reached, missing: {missing_sensors}")
        return False, missing_sensors

    def process_data(self, frame_id):
        """Process sensor data."""
        data_dict = {}

        # get camera and lidar poses
        data_dict['poses'] = {}
        camera_00_rgb = self.sensors.get('camera_00_rgb')
        if camera_00_rgb and camera_00_rgb.is_alive:
            data_dict['poses']['cam_00'] = camera_00_rgb.get_transform()
        lidar = self.sensors.get('lidar_ray_cast')
        if lidar and lidar.is_alive:
            data_dict['poses']['lidar'] = lidar.get_transform()

        # get traffic info
        data_dict['traffic'] = self.get_traffic_info()

        with self.data_lock:
            if frame_id in self.frame_data:
                for sensor_name, sensor_data in self.frame_data[frame_id].items():
                    data_dict[sensor_name] = sensor_data

                # Clean up processed data to save memory
                del self.frame_data[frame_id]
            else:
                logger.warning(f"No sensor data found for frame {frame_id}")

        return data_dict

    def get_traffic_info(self):
        """Get information about all traffic participants."""
        ego_vehicle = self.traffic_generator.ego_vehicle
        ego_location = ego_vehicle.get_location()
        max_distance = 80.0  # 80 meters

        traffic_info = {
            'ego_vehicles':
                {
                    'id': ego_vehicle.id,
                    'type': ego_vehicle.type_id,
                    'velocity': ego_vehicle.get_velocity(),
                    'acceleration': ego_vehicle.get_acceleration(),
                    'angular_velocity': ego_vehicle.get_angular_velocity(),
                },
            'vehicles': [],
            'pedestrians': []
        }

        if (not self.traffic_generator.vehicles_list) and (not self.traffic_generator.walkers_list):
            return traffic_info
        for actor_id in self.traffic_generator.vehicles_list + self.traffic_generator.walkers_list:
            actor = self.world.get_actor(actor_id)
            if actor is None or not actor.is_alive:
                continue

            distance = ego_location.distance(actor.get_location())
            if distance > max_distance:
                continue

            actor_info = {
                'id': actor_id,
                'type': actor.type_id,
                'transform': actor.get_transform().get_matrix(),
                'distance_to_ego': distance,  # Add distance information
            }

            if actor.type_id.startswith('vehicle.'):
                actor_info.update({
                    'velocity': actor.get_velocity(),
                    'bbox': actor.bounding_box,
                })
                traffic_info['vehicles'].append(actor_info)

            elif actor.type_id.startswith('walker.'):
                bones = {}
                for bone_transform in actor.get_bones().bone_transforms:
                    bone_name = bone_transform.name
                    if bone_name in ['crl_leg__L', 'crl_leg__R']:
                        bones[bone_name] = bone_transform.component.get_matrix()

                actor_info.update({
                    'bbox_extent': actor.bounding_box.extent,
                    'bones': bones
                })
                traffic_info['pedestrians'].append(actor_info)

        return traffic_info

    def update_spectator(self):
        spectator = self.world.get_spectator()
        transform = self.traffic_generator.ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(
            transform.location + carla.Location(z=50),
            carla.Rotation(pitch=-90)
        ))

    def cleanup_old_frame_data(self, current_frame, keep_frames=5):
        """Clean up old frame data to save memory"""
        with self.data_lock:
            frames_to_remove = [f for f in self.frame_data.keys()
                                if f < current_frame - keep_frames]
            for frame_id in frames_to_remove:
                del self.frame_data[frame_id]
                logger.info(f"Cleaned up data for frame {frame_id}")

    def collect_data(self, seq_id):
        """Main data collection loop."""
        self.traffic_light_controller.reset(permanent_green=(seq_id == 0))

        # Register sensor listeners
        self.register_sensor_listeners(self.sensors)

        logger.info("Warming up sensors...")
        for frame in range(5):
            carla_frame = self.world.tick()
            self.current_frame = carla_frame
            self.update_spectator()

            time.sleep(0.2)

            success, missing_sensors = self.wait_for_sensor_data(carla_frame)
            if success:
                data_dict = self.process_data(carla_frame)
                logger.info(f"Warming up sensors... [success] for frame {frame + 1}/5")
            else:
                logger.warning(f"Warming up sensors... [failed] for frame {frame + 1}/5")

        self.data_saver.make_dirs(f"{self.town_name}_Seq{seq_id:02d}")
        logger.info(f"Collecting {self.town_name}_Seq{seq_id:02d}...")

        # Statistics
        successful_frames = 0
        failed_frames = 0
        missing_data_count = defaultdict(int)

        # Collect data
        for frame in tqdm(range(self.config['frames']), desc=f"Sequence {seq_id}"):
            carla_frame = self.world.tick()
            self.current_frame = carla_frame
            self.update_spectator()

            self.traffic_light_controller.handle_traffic_lights(frame)

            success, missing_sensors = self.wait_for_sensor_data(carla_frame)
            if success:
                data_dict = self.process_data(carla_frame)
                self.data_saver.save_data(frame, data_dict)
                successful_frames += 1
            else:
                # Record missing sensors
                for sensor in missing_sensors:
                    missing_data_count[sensor] += 1
                failed_frames += 1

                # Try to save partial data
                data_dict = self.process_data(carla_frame)
                if data_dict:  # If there is partial data, still save it
                    self.data_saver.save_data(frame, data_dict)
                    logger.warning(f"Frame {frame}: Saved partial data, missing {missing_sensors}")

            # Periodically clean up old data
            if frame % 50 == 0:
                self.cleanup_old_frame_data(carla_frame)

        # Print statistics
        logger.info(f"\nSequence {seq_id} completed:")
        logger.info(f"  Successful frames: {successful_frames}/{self.config['frames']}")
        logger.info(f"  Failed frames: {failed_frames}")
        if missing_data_count:
            logger.info("  Missing data statistics:")
            for sensor, count in missing_data_count.items():
                logger.info(f"    {sensor}: {count} times")
