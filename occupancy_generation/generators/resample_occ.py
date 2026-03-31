from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from utils.occ_ops import occ_resample_sparse


@hydra.main(config_path="../config", config_name="resample", version_base=None)
def main(cfg: DictConfig) -> None:
    """Resample occupancy data from source to target configuration."""
    dataset_dir = Path(cfg.dataset_dir)

    # Source configuration
    src_voxel_size = float(cfg.source.voxel_size)
    src_voxel_origin = np.array(cfg.source.voxel_origin, dtype=np.float32)
    src_dir_name = cfg.source.save_dir_name

    # Target configuration
    tgt_voxel_size = float(cfg.target.voxel_size)
    tgt_voxel_origin = np.array(cfg.target.voxel_origin, dtype=np.float32)
    tgt_volume_size = np.array(cfg.target.volume_size, dtype=np.int32)
    tgt_dir_name = cfg.target.save_dir_name

    print("=" * 60)
    print(f"Dataset directory  : {dataset_dir}")
    print(f"Source voxel size  : {src_voxel_size} m")
    print(f"Source dir name    : {src_dir_name}")
    print(f"Target voxel size  : {tgt_voxel_size} m")
    print(f"Target dir name    : {tgt_dir_name}")
    print(f"Target voxel origin: {tgt_voxel_origin}")
    print(f"Target volume size : {tgt_volume_size}")
    print("=" * 60)

    for town_name in cfg.town_names:
        for sequence in cfg.sequences:
            sequence_name = f'{town_name}_Seq{sequence}'

            ori_dir = dataset_dir / sequence_name / 'occupancy' / src_dir_name
            tgt_dir = dataset_dir / sequence_name / 'occupancy' / tgt_dir_name
            tgt_dir.mkdir(parents=True, exist_ok=True)

            occ_paths = sorted(ori_dir.glob('*.npz'))

            for occ_path in tqdm(
                    occ_paths,
                    desc=f'[{sequence_name}] voxel size = {tgt_voxel_size}',
                    unit='frame'
            ):
                occ = np.load(occ_path)
                occupancy = occ['occupancy']

                occ_new = occ_resample_sparse(
                    occupancy,
                    src_voxel_size=src_voxel_size,
                    src_voxel_origin=src_voxel_origin,
                    tgt_voxel_size=tgt_voxel_size,
                    tgt_voxel_origin=tgt_voxel_origin,
                    tgt_volume_size=tgt_volume_size,
                    use_sem_prior=cfg.use_sem_prior
                )

                save_path = tgt_dir / f'{occ_path.stem}.npz'
                np.savez_compressed(
                    save_path,
                    occupancy=occ_new,
                    voxel_size=tgt_voxel_size,
                    voxel_origin=tgt_voxel_origin,
                    volume_size=tgt_volume_size,
                )


if __name__ == '__main__':
    main()
