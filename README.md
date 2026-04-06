<p align="center">
  <img src="docs/source/_static/carlaocc.png" alt="CarlaOcc Logo" width="60%">
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
  <img src="docs/source/_static/town10.gif" width="100%" />
</p>


## News

- 2026-04-03: Full **CarlaOccV1** is available on [Hugging Face](https://huggingface.co/datasets/fengyi233/CarlaOcc) (multi-modal archives; download and unpack as below).
- 2026-03-31: Code and mini dataset (**CarlaOccV1_mini**) released!
- 2026-02-23: Paper accepted by CVPR 2026!

## TODO
- [ ] Release training and evaluation code
- [x] Release full dataset (CarlaOccV1) — on [Hugging Face](https://huggingface.co/datasets/fengyi233/CarlaOcc)
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


## Download Dataset

**CarlaOccV1** (full) and **CarlaOccV1_mini** are hosted on [Hugging Face](https://huggingface.co/datasets/fengyi233/CarlaOcc). A [Baiduyun Disk](https://pan.baidu.com/s/1dfEJnxCHXKpTx2DbpX2nXQ?pwd=92pj) mirror is also listed in the [Dataset Tutorial](https://carlaocc.readthedocs.io/en/latest/dataset_tutorial.html).

**Mini set** (~3 GB, tutorial / quick visualization):

```bash
hf download fengyi233/CarlaOcc --include "CarlaOccV1_mini/*" --repo-type=dataset --local-dir .
```

**Full CarlaOccV1**:

```bash
hf download fengyi233/CarlaOcc --repo-type=dataset --local-dir .
```

Use `--include` / `--exclude` to fetch only some modalities (e.g. `CarlaOccV1/all_rgb/*`); see the [Dataset Tutorial](https://carlaocc.readthedocs.io/en/latest/dataset_tutorial.html).

After downloading, restore the on-disk tree:

```bash
# Extract under ./CarlaOccV1 (skip if you only use CarlaOccV1_mini and did not download archives)
bash scripts/unzip_dataset.sh CarlaOccV1
```

You can also use `bash scripts/unzip_dataset.sh <download_dir> [output_dir]`; see `scripts/unzip_dataset.sh`.


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

Full documentation is on [Read the Docs](https://carlaocc.readthedocs.io/en/latest/), including the [Dataset Tutorial](https://carlaocc.readthedocs.io/en/latest/dataset_tutorial.html) (download, layout, Baiduyun link).


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
