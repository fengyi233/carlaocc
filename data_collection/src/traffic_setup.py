import logging
import random
import time
from typing import List, Dict

import carla
import numpy as np

logger = logging.getLogger(__name__)


class TrafficGenerator:
    def __init__(self, world: carla.World, client: carla.Client, config: Dict):
        self.world = world
        self.client = client
        self.config = config

        self.traffic_manager = self.client.get_trafficmanager(self.config['tm_port'])
        self.setup_traffic_manager()

        self.ego_vehicle = None
        self.ego_vehicle_spawn_index = None
        self.vehicles_list = []
        self.walkers_list = []
        self.controllers_list = []

    def setup_traffic_manager(self):
        # Set traffic manager parameters
        self.traffic_manager.set_global_distance_to_leading_vehicle(self.config['global_distance_to_leading_vehicle'])
        self.traffic_manager.global_percentage_speed_difference(self.config['global_percentage_speed_difference'])

        # Set seed
        seed = self.config['seed'] if self.config['seed'] is not None else int(time.time())
        self.world.set_pedestrians_seed(seed)
        self.traffic_manager.set_random_device_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Set synchronous mode
        self.traffic_manager.set_synchronous_mode(True)

    def generate_traffic(self, seq_count: int = 0):
        # Spawn ego vehicle first
        if self.ego_vehicle is None:
            self.spawn_ego_vehicle(seq_count)

        # Calculate actual vehicle and walker numbers based on road length
        # vehicle_num and walker_num in config are per kilometer
        vehicle_num_per_km = self.config.get('vehicle_num', 0)
        walker_num_per_km = self.config.get('walker_num', 0)

        road_length_km = self.config['map_road_lengths'][self.config['map_name']]
        max_vehicle_spawn_num = int(self.config['max_vehicle_spawn_num'])
        max_walker_spawn_num = int(self.config['max_walker_spawn_num'])

        spawn_vehicle_num = min(int(vehicle_num_per_km * road_length_km), max_vehicle_spawn_num)
        spawn_walker_num = min(int(walker_num_per_km * road_length_km), max_walker_spawn_num)

        # Spawn vehicles
        if spawn_vehicle_num > 0:
            self.vehicles_list = self._spawn_vehicles(spawn_vehicle_num)

        # Spawn walkers
        if spawn_walker_num > 0:
            self.controllers_list, self.walkers_list = self._spawn_walkers(spawn_walker_num)

    def _farthest_point_sampling(self, spawn_points: List[carla.Transform], num_samples: int) -> List[int]:
        """
        Use Farthest Point Sampling algorithm to select spatially well-distributed spawn points.
        
        Args:
            spawn_points: All available spawn points
            num_samples: Number of points to sample
            
        Returns:
            List of indices of sampled points
        """
        num_points = len(spawn_points)
        if num_samples >= num_points:
            return list(range(num_points))

        # Extract coordinates (x, y, z) of all points
        coords = np.array([[sp.location.x, sp.location.y, sp.location.z]
                           for sp in spawn_points])

        # Initialize: select the first point (use fixed index for reproducibility)
        selected_indices = [0]

        # Calculate minimum distance from all points to selected points
        min_distances = np.full(num_points, np.inf)

        for _ in range(num_samples - 1):
            # Update distances to the latest selected point
            last_selected = selected_indices[-1]
            distances = np.linalg.norm(coords - coords[last_selected], axis=1)
            min_distances = np.minimum(min_distances, distances)

            # Set distance of selected points to 0 to avoid re-selection
            min_distances[selected_indices] = 0

            # Select the point with maximum distance
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)

        return selected_indices

    def spawn_ego_vehicle(self, seq_count: int = 0) -> carla.Actor:
        """Spawn the ego vehicle using Farthest Point Sampling for spatially uniform distribution."""

        # Get all spawn points
        all_spawn_points = self.world.get_map().get_spawn_points()
        num_spawn_points = len(all_spawn_points)

        # Use Farthest Point Sampling to select spatially well-distributed points
        num_samples = min(self.config['seqs_per_map'], num_spawn_points)
        sampled_indices = self._farthest_point_sampling(all_spawn_points, num_samples)

        # Select spawn point based on seq_count
        if seq_count >= len(sampled_indices):
            logger.warning(
                f"seq_count {seq_count} exceeds sampled indices length {len(sampled_indices)}, using last index")
            seq_count = len(sampled_indices) - 1

        spawn_index = sampled_indices[seq_count]
        spawn_point = all_spawn_points[spawn_index]

        vehicle_bp = self.world.get_blueprint_library().filter(self.config['ego_vehicle']['name'])[0]
        if vehicle_bp.has_attribute('color'):
            color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
            vehicle_bp.set_attribute('color', color)

        # Spawn ego vehicle
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.ego_vehicle_spawn_index = spawn_index

        # Configure ego vehicle behavior
        self.traffic_manager.ignore_lights_percentage(self.ego_vehicle, self.config['ignore_lights_percentage'])
        self.traffic_manager.ignore_signs_percentage(self.ego_vehicle, self.config['ignore_signs_percentage'])
        self.traffic_manager.ignore_vehicles_percentage(self.ego_vehicle, self.config['ignore_vehicles_percentage'])
        self.traffic_manager.ignore_walkers_percentage(self.ego_vehicle, self.config['ignore_walkers_percentage'])
        self.traffic_manager.vehicle_percentage_speed_difference(self.ego_vehicle,
                                                                 self.config['ego_vehicle'][
                                                                     'vehicle_percentage_speed_difference'])
        self.ego_vehicle.set_autopilot(True, self.traffic_manager.get_port())

        return self.ego_vehicle

    def _spawn_vehicles(self, vehicle_num: int) -> List[int]:
        """Spawn vehicles in the world, avoiding ego vehicle's spawn point."""
        # Get available spawn points
        all_spawn_points = self.world.get_map().get_spawn_points()
        all_spawn_points.pop(self.ego_vehicle_spawn_index)
        spawn_points_num = len(all_spawn_points)
        if vehicle_num < spawn_points_num:
            random.shuffle(all_spawn_points)
        elif vehicle_num > spawn_points_num:
            logger.warning(f"Requested {vehicle_num} vehicles, but could only find {spawn_points_num} spawn points,"
                           f" spawning {spawn_points_num} vehicles.")
            vehicle_num = spawn_points_num

        # Get vehicle blueprints
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [bp for bp in blueprints if bp.id not in self.config['excluded_vehicles']]
        current_map = self.world.get_map().name.split('/')[-1]
        if current_map in self.config['no_truck_maps']:
            blueprints = [bp for bp in blueprints if bp.id not in ['vehicle.fuso.mitsubishi',
                                                                   'vehicle.firetruck.actors',
                                                                   'vehicle.carlacola.actors',
                                                                   'vehicle.ambulance.ford']]

        vehicles_list = []
        batch = []
        for i in range(vehicle_num):
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)

            spawn_point = random.choice(all_spawn_points)
            all_spawn_points.remove(spawn_point)

            batch.append(carla.command.SpawnActor(blueprint, spawn_point)
                         .then(carla.command.SetAutopilot(carla.command.FutureActor,
                                                          True, self.traffic_manager.get_port())))

        # Execute batch
        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                while len(all_spawn_points) > 0:
                    # Retry with remaining spawn points
                    blueprint = random.choice(blueprints)
                    spawn_point = random.choice(all_spawn_points)
                    all_spawn_points.remove(spawn_point)
                    batch = [carla.command.SpawnActor(blueprint, spawn_point)
                             .then(carla.command.SetAutopilot(carla.command.FutureActor,
                                                              True, self.traffic_manager.get_port()))]
                    new_response = self.client.apply_batch_sync(batch, True)[-1]
                    if not new_response.error:
                        vehicles_list.append(new_response.actor_id)
                        break
            else:
                vehicles_list.append(response.actor_id)

        # Configure vehicle behavior for all spawned vehicles
        for vehicle_id in vehicles_list:
            vehicle = self.world.get_actor(vehicle_id)
            if vehicle and vehicle.is_alive:
                self.traffic_manager.ignore_walkers_percentage(vehicle, self.config['ignore_walkers_percentage'])
                self.traffic_manager.ignore_vehicles_percentage(vehicle, self.config['ignore_vehicles_percentage'])
                self.traffic_manager.ignore_lights_percentage(vehicle, self.config['ignore_lights_percentage'])
                self.traffic_manager.ignore_signs_percentage(vehicle, self.config['ignore_signs_percentage'])

        logger.info(f"Spawned {len(vehicles_list)} vehicles...")
        return vehicles_list

    def _spawn_walkers(self, number_of_walkers: int) -> "tuple[List[int], List[int]]":
        """Spawn walkers in the world."""

        bp_walkers = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        bp_walkers = sorted(bp_walkers, key=lambda bp: bp.id)

        walkers_list = []
        controllers_list = []

        # Get spawn points
        spawn_points = []
        for _ in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_point.location.z += 2  # Avoid collision spawning
                spawn_points.append(spawn_point)

        # Spawn walkers
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(bp_walkers)

            # Set walker attributes
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            if walker_bp.has_attribute('speed'):
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])  # walking
            else:
                logger.warning("Walker has no speed attribute")
                walker_speed.append(0.0)

            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        # Execute batch
        results = self.client.apply_batch_sync(batch, True)
        for i, response in enumerate(results):
            if response.error:
                # Respawn walker if failed
                while True:
                    # Retry with remaining spawn points
                    spawn_point = carla.Transform()
                    loc = self.world.get_random_location_from_navigation()
                    if loc is not None:
                        spawn_point.location = loc
                        spawn_point.location.z += 2

                    batch = [carla.command.SpawnActor(walker_bp, spawn_point)]
                    new_response = self.client.apply_batch_sync(batch, True)[-1]
                    if not new_response.error:
                        walkers_list.append(new_response.actor_id)
                        break
            else:
                walkers_list.append(response.actor_id)

        # Spawn walker controllers
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for walker_id in walkers_list:
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walker_id))

        results = self.client.apply_batch_sync(batch, True)
        for i, response in enumerate(results):
            if response.error:
                logger.error(f"Failed to spawn walker controller: {response.error}")
            else:
                controllers_list.append(response.actor_id)

        # Initialize walker controllers
        self.world.set_pedestrians_cross_factor(0.0)  # percentagePedestriansCrossing

        for i, controller_id in enumerate(controllers_list):
            controller = self.world.get_actor(controller_id)
            if controller and controller.is_alive:
                # Start walker
                controller.start()
                # Set walk to random point
                controller.go_to_location(self.world.get_random_location_from_navigation())
                # Set max speed
                controller.set_max_speed(float(walker_speed[i]))

        logger.info(f"Spawned {len(walkers_list)} walkers...")
        return controllers_list, walkers_list

    def destroy(self):
        """Destroy all spawned actors safely."""
        try:
            if self.ego_vehicle is not None and self.ego_vehicle.is_alive:
                self.ego_vehicle.destroy()
                logger.info("Ego vehicle destroyed.")
        except Exception as e:
            logger.warning(f"Failed to destroy ego vehicle: {e}")

        if self.vehicles_list:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
            self.vehicles_list = []
        if self.controllers_list or self.walkers_list:
            # 1. Stop all controllers
            for controller_id in self.controllers_list:
                controller = self.world.get_actor(controller_id)
                if controller and controller.is_alive:
                    controller.stop()

            # 2. Destroy controllers
            if self.controllers_list:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.controllers_list])
                self.controllers_list = []

            # 3. Destroy walker entities
            if self.walkers_list:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walkers_list])
                self.walkers_list = []

        for _ in range(2):
            self.world.tick()
            time.sleep(0.5)
        logger.info("Traffic destroyed.")


class TrafficLightController:
    def __init__(self, world: carla.World, ego_vehicle: carla.Actor, config):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.config = config
        self.start_wait_frame = None

    def reset(self, permanent_green: bool = False):
        """Reset traffic lights to default state."""
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        for light in traffic_lights:
            if permanent_green:
                light.set_state(carla.TrafficLightState.Green)
                light.freeze(True)
            else:
                light.freeze(False)
                light.reset_group()
                light.set_green_time(self.config['green_time'])
                light.set_yellow_time(self.config['yellow_time'])
                light.set_red_time(self.config['red_time'])

    def handle_traffic_lights(self, frame: int):
        if self.ego_vehicle.get_velocity().length() > self.config['stop_speed']:
            return
        traffic_light = self.ego_vehicle.get_traffic_light()
        if not traffic_light or traffic_light.get_state() != carla.TrafficLightState.Red:
            return

        if self.start_wait_frame is None:
            self.start_wait_frame = frame

        waited_frames = frame - self.start_wait_frame
        if waited_frames > self.config['max_wait_frames']:
            self.start_wait_frame = None
            traffic_light_group = traffic_light.get_group_traffic_lights()
            traffic_light_group = [light for light in traffic_light_group if
                                   light.id != traffic_light.id]
            for light in traffic_light_group:
                light.set_state(carla.TrafficLightState.Red)
            traffic_light.set_state(carla.TrafficLightState.Green)
