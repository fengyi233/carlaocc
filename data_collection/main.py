import logging
import time

import carla
import hydra
from omegaconf import DictConfig

from src.data_collector import DataCollector
from src.sensor_setup import SynchronousSensors
from src.traffic_setup import TrafficGenerator

# Configure logger
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    data_collection_config = cfg.data_collection
    traffic_config = cfg.traffic
    sensor_config = cfg.sensors.sensor_config

    world = None
    sensors = None
    traffic_generator = None

    try:
        # Connect to CARLA server
        client = carla.Client(cfg.host, cfg.port)
        client.set_timeout(60.0)
        available_maps = client.get_available_maps()
        logger.info(f"Available maps: {available_maps}")
        maps_to_collect = data_collection_config.maps_to_collect
        assert all(map_name in available_maps for map_name in
                   maps_to_collect), f"Not all maps are available: {maps_to_collect}"

        for map_name in maps_to_collect:
            world = client.get_world()
            current_map = world.get_map().name.split('/')[-1]
            if current_map != map_name:
                client.load_world(map_name)
                logger.info(f"loading Map {map_name} ... ")
                time.sleep(60)
                world = client.get_world()
            logger.info(f"Map {map_name} loaded")
            original_settings = world.get_settings()

            # Set synchronous mode
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / data_collection_config.fps
            world.apply_settings(settings)
            seqs_to_collect = data_collection_config.seqs_to_collect

            for seq in seqs_to_collect:
                seq_id = seq.id
                traffic_level = seq.traffic_level
                logger.info(f"\n---{map_name} Seq{seq_id:02d} ({traffic_level}) ---")

                seq_traffic_config = {
                    **traffic_config.common,
                    **traffic_config.levels[traffic_level],
                    'map_name': map_name,
                }
                traffic_generator = TrafficGenerator(world, client, seq_traffic_config)
                traffic_generator.generate_traffic(seq_id)

                sensors = SynchronousSensors(world, traffic_generator.ego_vehicle, sensor_config)
                data_collector = DataCollector(world, sensors.sensors, traffic_generator, data_collection_config)
                data_collector.collect_data(seq_id)

                if sensors is not None:
                    sensors.destroy()
                if traffic_generator is not None:
                    traffic_generator.destroy()

    except KeyboardInterrupt:
        logger.warning("Data collection interrupted by user")
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        raise
    finally:
        # Clean up
        logger.info("Cleaning up...")

        if world is not None:
            # Destroy traffic
            if traffic_generator is not None:
                traffic_generator.destroy()

            # Destroy sensors
            if sensors is not None:
                sensors.destroy()

            # Avoid abnormal exit
            for _ in range(2):
                world.tick()
                time.sleep(0.5)

            # Reset synchronous mode
            world.apply_settings(original_settings)
        logger.info("All actors destroyed.")


if __name__ == '__main__':
    main()
