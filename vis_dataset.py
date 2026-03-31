import argparse
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.load_utils import load_lidar, load_semantic_lidar
from utils.vis_utils import (
    vis_pano_occ,
    vis_pc,
    CameraSetting, CameraSettingConfig,
)

_VIS_MODALITY_CHOICES = [
    'rgb', 'depth', 'normal', 'semantics',
    'lidar', 'semantic_lidar',
    'occupancy/vs_0_1',
    'occupancy/vs_0_2_forward_view',
    'occupancy/vs_0_4_surround_view',
    'all',
]


def vis_multi_img(seq_dir, frame_id, modality_type='rgb', save_path=None):
    """Visualize all 6 cameras for a given modality in a 3×2 grid.

    Layout::

        cam_00  cam_01
        cam_02  cam_03
        cam_04  cam_05

    Args:
        seq_dir: Sequence directory (Path).
        frame_id: Integer frame index.
        modality_type: One of ``'rgb'``, ``'depth'``, ``'normal'``, ``'semantics'``.
        save_path: Save figure to this path; display interactively if ``None``.
    """
    frame_str = f"{frame_id:04d}"
    cam_ids = ['00', '01', '02', '03', '04', '05']
    is_depth = modality_type == 'depth'

    fig, axes = plt.subplots(3, 2, figsize=(20, 10))

    for idx, cam_id in enumerate(cam_ids):
        row, col = idx // 2, idx % 2
        ax = axes[row][col]

        path = seq_dir / modality_type / f'image_{cam_id}' / f'{frame_str}.png'

        if is_depth:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                ax.set_visible(False)
                continue
            depth_m = img.astype(np.float32) / 65535.0 * 80.0
            ax.imshow(depth_m, cmap='plasma', vmin=0, vmax=80)
        else:
            img = cv2.imread(str(path))
            if img is None:
                ax.set_visible(False)
                continue
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ax.set_title(f'cam {cam_id}')
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def vis_modality(modality, seq_dir, frame_id, save_dir=None):
    """Visualize a specific modality for a given frame."""
    frame_str = f"{frame_id:04d}"

    def get_save_path(name):
        if save_dir is None:
            return None
        return str(Path(save_dir) / f'{name}_{frame_str}.png')

    if modality in ('rgb', 'depth', 'normal', 'semantics'):
        vis_multi_img(seq_dir, frame_id, modality_type=modality,
                      save_path=get_save_path(modality))

    elif modality in ('lidar', 'semantic_lidar'):
        path = seq_dir / modality / f'{frame_str}.ply'
        loader = load_semantic_lidar if modality == 'semantic_lidar' else load_lidar
        pc = loader(path)
        if pc is not None:
            vis_pc(pc, pc_type=modality, save_path=get_save_path(modality))

    elif modality.startswith('occupancy/'):
        occ_subdir = modality.split('/', 1)[1]
        path = seq_dir / 'occupancy' / occ_subdir / f'{frame_str}.npz'
        if not path.exists():
            print(f"  Skipping {modality} — file not found: {path}")
            return
        data = np.load(path)
        occupancy, voxel_size = data['occupancy'], float(data['voxel_size'])
        cam_setting = CameraSetting.from_config(CameraSettingConfig(
            cam_position=(0, 25.6, 30),
            cam_pitch=45,
        ))
        vis_pano_occ(occupancy, voxel_size=voxel_size,
                     camera_setting=cam_setting, show_axis=True,
                     save_path=get_save_path(f'occupancy_{occ_subdir}'))

    else:
        print(f"Unknown modality: {modality}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Visualize CarlaOcc dataset modalities for a given sequence and frame.')
    ap.add_argument("--dataset_dir", type=str, default="data/CarlaOccV1_mini",
                    help="Root directory of the dataset")
    ap.add_argument("--towns", type=str, default="Town01",
                    choices=["Town01", "Town02", "Town03", "Town04",
                             "Town05", "Town06", "Town07", "Town10HD"])
    ap.add_argument("--seqs", type=str, default="Seq07",
                    choices=[f"Seq{i:02d}" for i in range(13)])
    ap.add_argument("--frame_id", type=str, default='0',
                    help="Frame index (integer) or 'all' for every frame in the sequence")
    ap.add_argument("--vis_modality", default='all',
                    choices=_VIS_MODALITY_CHOICES,
                    help="Modality to visualize, or 'all' for every modality")
    ap.add_argument("--interactive", action='store_true',
                    help="Display visualizations interactively instead of saving to disk")
    ap.add_argument("--save_dir", type=str, default='outputs/vis',
                    help="Directory for saved outputs (ignored when --interactive is set)")

    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    seq_name = f"{args.towns}_Opt_{args.seqs}"
    seq_dir = dataset_dir / seq_name
    save_dir = None if args.interactive else str(Path(args.save_dir) / seq_name)

    print(f"Dataset : {dataset_dir}")
    print(f"Sequence: {seq_name}")
    print(f"Frame   : {args.frame_id}")
    print(f"Modality: {args.vis_modality}")
    print("-" * 60)

    if not seq_dir.exists():
        print(f"Error: sequence directory not found: {seq_dir}")
        exit(1)

    vis_modalities = (
        [c for c in _VIS_MODALITY_CHOICES if c != 'all']
        if args.vis_modality == 'all'
        else [args.vis_modality]
    )

    if args.frame_id == 'all':
        rgb_dir = seq_dir / 'rgb' / 'image_00'
        if rgb_dir.exists():
            frame_ids = sorted(int(p.stem) for p in rgb_dir.glob('*.png'))
        else:
            print(f"Warning: cannot discover frames from {rgb_dir}, defaulting to 0–999")
            frame_ids = list(range(1000))
    else:
        frame_ids = [int(args.frame_id)]

    for frame_id in tqdm(frame_ids, desc='Processing frames'):
        for mod in vis_modalities:
            vis_modality(mod, seq_dir, frame_id, save_dir=save_dir)
