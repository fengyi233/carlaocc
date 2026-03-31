<p align="center">
  <img src="docs/_static/carlaocc.png" alt="CarlaOcc Logo" width="60%">
</p>

<h2 align="left"><b>An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving</b></h2>


<p align="center">
  <a href="https://mias.group/CarlaOcc" style="text-decoration:none;">
      <strong><img src="https://img.icons8.com/color/48/home.png" alt="Home" width="20" height="20" style="vertical-align:middle; margin-right:4px;"/>Homepage</strong>
  </a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://www.youtube.com/watch?v=b8nFA7ydC_A" style="text-decoration:none;">
      <strong><img src="https://img.icons8.com/color/48/youtube-play.png" alt="YouTube" width="20" height="20" style="vertical-align:middle; margin-right:4px;"/>Video</strong>
  </a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2603.27238" style="text-decoration:none;">
      <strong><img src="https://static.arxiv.org/static/browse/0.3.4/images/icons/favicon-32x32.png" alt="arXiv" width="20" height="20" style="vertical-align:middle; margin-right:4px;"/>Paper</strong>
  </a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/fengyi233/CarlaOcc" style="text-decoration:none;">
      <strong><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="20" height="20" style="vertical-align:middle; margin-right:4px;"/>Dataset</strong>
  </a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://carlaocc.readthedocs.io/en/latest/" style="text-decoration:none;">
    <strong><img src="https://img.icons8.com/color/48/book.png" alt="docs" width="20" height="20" style="vertical-align:middle; margin-right:4px;"/>Docs</strong>
  </a>
</p>

<p align="center">
  <video src="docs/_static/town10.mp4" autoplay loop muted playsinline width="100%"></video>
</p>


## News

- 2026-03-31: Code and Dataset released!
- 2026-02-23: Paper accepted by CVPR 2026!

## TODO
- [ ] Release training and evaluation code
- [ ] Release full dataset (CarlaOccV1)
- [x] Release mini dataset (CarlaOccV1_mini)
- [x] Release dataset tutorial and visualization tools
- [x] Release data collection, scene exportation, & occupancy generation pipeline

## Highlights

- **Panoptic Occupancy Ground Truth** — 100K frames with voxel-level semantic + instance labels, resolution up to **0.05 m**
- **Data Modality** — 6 RGB cameras (KITTI-360-style rig + 2 rear-view cameras) with depth, semantic, surface normal images, semantic lidar and panoptic occupancy
- **Occupancy Consistency** — Physically consistent occupancy ground truth generated via high-quality mesh voxelization
- **Scene Meshes** — Exportable foreground & background actor meshes with per-actor semantic labels

## Requirements

- **CARLA UE5 server** (v0.10.0) — [Build instructions](https://carla-ue5.readthedocs.io/en/latest/#getting-started)
- **Python 3.8+**
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  pip install <carla_wheel>.whl   # matching your CARLA build
  ```


## Pipeline Overview

CarlaOcc provides the complete toolchain from data collection to occupancy ground truth generation:

```text
+---------------------------------------+
| 1. Data Collection (CARLA Simulator)  |
+---------------------------------------+
                   |
                   v
+---------------------------------------+
| 2. Scene Exportation (UE5 + Python)   |
+---------------------------------------+
                   |
                   v
+---------------------------------------+
| 3. Occupancy Generation (Voxelization)|
+---------------------------------------+
```
### 1. Data Collection (`data_collection/`)

Runs a synchronous CARLA session that spawns an ego vehicle with configurable traffic, and records multi-modal sensor data to disk.

```bash
cd data_collection/
python main.py                                  # default config
python main.py data_collection.frames=100       # override via Hydra
```

### 2. Scene Exportation (`scene_exportation/`)

Exports static scene geometry and semantic annotations from the UE5 editor, then post-processes meshes in Python.

**Step 1** — Export actors from UE5 editor:
```python
# In UE5 Python console
path/to/your/script/export_scene.py
```

**Step 2** — Reconstruct & voxelize meshes:
```bash
cd scene_exportation/
python recon_actor.py
# Override via Hydra, e.g.:
python recon_actor.py towns.Town01_Opt.region.min=[-100,-100,-20] voxelization.bg_voxel_size=0.1
```

### 3. Occupancy Generation (`occupancy_generation/`)

Generates dense panoptic occupancy ground truth by compositing foreground and background voxelizations.

```bash
# Generate semantic & depth (with artifact rectification)
python occupancy_generation/gen_sem_depth.py

# Generate panoptic occupancy labels
python occupancy_generation/gen_pano_occ.py

# Generate surface normals
python occupancy_generation/gen_normal.py
```

## Visualization

### Quick Dataset Visualization

`vis_dataset.py` is the top-level script for visualizing any modality for a given sequence and frame:

```bash
# Single frame, single modality
python vis_dataset.py --dataset_dir data/CarlaOccV1_mini \
    --towns Town01 --seqs Seq07 --frame_id 0 --vis_modality rgb

# All modalities for one frame (saves to outputs/vis/ by default)
python vis_dataset.py --dataset_dir data/CarlaOccV1_mini \
    --towns Town01 --seqs Seq07 --frame_id 0 --vis_modality all

# Interactive display (no saving)
python vis_dataset.py --dataset_dir data/CarlaOccV1_mini \
    --towns Town01 --seqs Seq07 --frame_id 0 --interactive

# All frames in the sequence
python vis_dataset.py --dataset_dir data/CarlaOccV1_mini \
    --towns Town01 --seqs Seq07 --frame_id all --vis_modality rgb
```

**Available modalities:** `rgb`, `depth`, `normal`, `semantics`, `lidar`, `semantic_lidar`, `occupancy/vs_0_1`, `occupancy/vs_0_2_forward_view`, `occupancy/vs_0_4_surround_view`

### Pipeline Visualization

```bash
# Visualize scene mesh
python occupancy_generation/visualizers/vis_mesh.py

# Visualize occupancy grid
python occupancy_generation/visualizers/vis_occ.py
```

## Documentation

Full documentation is available on [Read the Docs](https://carlaocc.readthedocs.io/en/latest/).


## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{feng2026carlaocc,
  title={An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving},
  author={Feng, Yi and E, Junwu and Guo, Zizhan and Ma, Yu and Wang, Hanli and Fan, Rui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
