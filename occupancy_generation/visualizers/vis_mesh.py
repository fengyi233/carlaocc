from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from occupancy_generation.generators.mesh_generator import SceneMeshGenerator
from utils.vis_utils import CameraSetting, CameraSettingConfig, vis_pano_mesh, vis_sem_mesh


@hydra.main(config_path="../config", config_name="vis_mesh", version_base=None)
def main(cfg: DictConfig) -> None:
    mesh_type = cfg.mesh_type
    save_dir = Path(cfg.save_dir)
    bbox_enabled = cfg.get("vis_bbox", True)
    vis_interval = int(cfg.get("vis_interval", 1))
    if vis_interval < 1:
        raise ValueError(f"vis_interval must be >= 1, got {vis_interval}")

    camera_setting = CameraSetting.from_config(
        CameraSettingConfig(
            image_size=cfg.vis.image_size,
            focal_length=cfg.vis.focal_length,
            cam_position=cfg.vis.cam_position,
            cam_pitch=cfg.vis.cam_pitch,
        )
    ) if cfg.get("vis") else None

    print(f"Frame range  : [{cfg.frame_range[0]}, {cfg.frame_range[1]}]")
    print(f"Vis interval : every {vis_interval} frame(s)")
    for town_name in cfg.town_names:
        for sequence in cfg.sequences:
            print(f"\nVisualizing {mesh_type} mesh: {town_name} Seq{sequence}")
            mesh_generator = SceneMeshGenerator(cfg, town_name, sequence)
            bbox = mesh_generator.occ_bbox if bbox_enabled else None

            seq_save_dir = save_dir / mesh_type / f"{town_name}_Seq{sequence}"
            seq_save_dir.mkdir(parents=True, exist_ok=True)

            for i in tqdm(
                    range(cfg.frame_range[0], cfg.frame_range[1] + 1, vis_interval),
                    desc=f"{town_name}_Seq{sequence}",
            ):
                save_path = str(seq_save_dir / f"{i:04d}.png")
                if mesh_type == "semantic":
                    mesh = mesh_generator.generate_sem_mesh(i)
                    vis_sem_mesh(mesh,
                                 bbox=bbox,
                                 camera_setting=camera_setting,
                                 save_path=save_path)
                elif mesh_type == "panoptic":
                    mesh = mesh_generator.generate_pano_mesh(i)
                    vis_pano_mesh(mesh,
                                  bbox=bbox,
                                  camera_setting=camera_setting,
                                  save_path=save_path)
                else:
                    raise ValueError(f"Unknown mesh type: {mesh_type}")


if __name__ == '__main__':
    main()
