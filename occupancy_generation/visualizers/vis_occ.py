from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from utils.vis_utils import vis_pano_occ, CameraSetting, CameraSettingConfig


@hydra.main(config_path="../config", config_name="vis_occ", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset_dir = Path(cfg.dataset_dir)
    occ_dir_name = cfg.occ_dir_name
    save_dir = Path(cfg.save_dir)
    voxel_origin = np.array(cfg.voxel_origin, dtype=np.float32)

    # Camera setting
    camera_setting = CameraSetting.from_config(CameraSettingConfig(
        image_size=cfg.camera.image_size,
        focal_length=float(cfg.camera.focal_length),
        cam_position=cfg.camera.cam_position - voxel_origin,
        cam_pitch=float(cfg.camera.cam_pitch),
    ))

    sequences = cfg.sequences
    frame_range = cfg.frame_range
    vis_interval = int(cfg.get("vis_interval", 1))
    if vis_interval < 1:
        raise ValueError(f"vis_interval must be >= 1, got {vis_interval}")
    print("=" * 60)
    print("Visualizing occupancy ground truth...")
    print(f"Dataset directory  : {dataset_dir}")
    print(f"Occ directory name : {occ_dir_name}")
    print(f"Save directory     : {save_dir}")
    print(f"Voxel origin       : {voxel_origin}")
    print(f"Frame range        : {frame_range if frame_range is not None else 'all'}")
    print(f"Vis interval       : every {vis_interval} frame(s)")
    print("=" * 60)

    for town_name in cfg.town_names:
        for sequence in sequences:
            sequence_name = f'{town_name}_Seq{sequence}'

            occ_dir = dataset_dir / sequence_name / 'occupancy' / occ_dir_name

            occ_paths = sorted(occ_dir.glob("*.npz"))
            if frame_range is not None:
                start, end = frame_range
                occ_paths = occ_paths[start:end + 1]
            if vis_interval > 1:
                occ_paths = occ_paths[::vis_interval]
            for occ_path in tqdm(
                    occ_paths,
                    desc=f'[{sequence_name}] Visualization',
                    unit='frame'
            ):
                occ = np.load(occ_path)
                occupancy = occ['occupancy']
                voxel_size = occ['voxel_size']

                seq_save_dir = save_dir / sequence_name
                seq_save_dir.mkdir(parents=True, exist_ok=True)
                save_path = (seq_save_dir / f'{occ_path.stem}.png').as_posix()

                vis_pano_occ(occupancy,
                             voxel_size=voxel_size,
                             save_path=save_path,
                             colormap='carla',
                             camera_setting=camera_setting)


if __name__ == '__main__':
    main()
