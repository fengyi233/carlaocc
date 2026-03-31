from pathlib import Path
from typing import Dict

import cv2
import d2nt
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm


class NormalGenerator:
    """Generate surface normal maps from depth images."""

    def __init__(self, cfg: DictConfig, town_name: str, sequence: str):
        self.cfg = cfg
        dataset_dir = Path(cfg.dataset_dir)
        self.data_dir = dataset_dir / f"{town_name}_Seq{sequence}"
        self.intrinsic_matrix = np.array(cfg.intrinsic_matrix, dtype=np.float32)

    def gen_normal(self, frame_id: int) -> Dict[str, np.ndarray]:
        """Generate normal maps for all cameras.

        Args:
            frame_id: Frame number to process

        Returns:
            Dict mapping camera ID to normal image (uint8 BGR, shape H×W×3)
        """
        results = {}
        for cam_id in ['00', '01', '02', '03', '04', '05']:
            # Load refined depth map
            refined_depth_path = self.data_dir / "depth" / f'image_{cam_id}' / f"{frame_id:04d}.png"
            refined_depth = cv2.imread(str(refined_depth_path), -1)
            refined_depth = refined_depth.astype(np.float32) / 65535 * 80.0

            normal_map = d2nt.depth2normal(refined_depth, self.intrinsic_matrix)
            normal_img = (d2nt.get_normal_vis(normal_map) * 255).astype(np.uint8)
            results[cam_id] = cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR)
        return results


@hydra.main(config_path="../config", config_name="gen_normal", version_base=None)
def main(cfg: DictConfig) -> None:
    for town_name in cfg.town_names:
        for sequence in cfg.sequences:
            print(f"\nProcessing {town_name} Seq{sequence}")
            generator = NormalGenerator(cfg, town_name, sequence)

            for i in tqdm(range(cfg.frame_range[0], cfg.frame_range[1] + 1),
                          desc=f"{town_name}_Seq{sequence}"):
                normals = generator.gen_normal(i)
                for cam_id, normal_img in normals.items():
                    save_path = generator.data_dir / "normal" / f'image_{cam_id}' / f"{i:04d}.png"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_path), normal_img)


if __name__ == '__main__':
    main()
