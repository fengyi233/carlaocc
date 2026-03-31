Dataset Tutorial
==========

Prepare Environment
-------------------

Create Python environment using conda:

.. code-block:: bash

    conda create -n carlaocc python=3.10 -y
    conda activate carlaocc
    pip install -r requirements.txt

Prepare Dataset
---------------

Our dataset is released on `HuggingFace <https://huggingface.co/datasets/fengyi233/CarlaOcc>`_ and
`Baiduyun Disk <https://pan.baidu.com/s/1dfEJnxCHXKpTx2DbpX2nXQ?pwd=92pj>`_.

- We provide a CarlaOccV1_mini (~3GB) for visualization and tutorial. To download it, run:

.. code-block:: bash

    hf download fengyi233/CarlaOcc --include "CarlaOccV1_mini/*" --repo-type=dataset --local-dir .

- To download the whole dataset, run:

.. code-block:: bash

    hf download fengyi233/CarlaOcc --repo-type=dataset --local-dir .

- To download only specific data modalities, use ``--include`` or ``--exclude``:

.. code-block:: bash

    # Only RGB images
    hf download fengyi233/CarlaOcc --include "CarlaOccV1/all_rgb/*" --repo-type=dataset --local-dir .

    # Exclude semantic_lidar and semantic maps
    hf download fengyi233/CarlaOcc --exclude "CarlaOccV1/all_semantic_lidar/*" "CarlaOccV1/all_semantics/*" --repo-type=dataset --local-dir .

After downloading, unzip the data using:

.. code-block:: bash

    # Skip this if only miniset downloaded
    # By default, this will extract files in the ./CarlaOccV1 directory
    bash scripts/unzip_dataset.sh CarlaOccV1


The CarlaOcc dataset is structured as follows:

.. code-block:: console

   CarlaOccV1/
   |-- calib/
   |   └── calib.yaml
   |-- splits/
   |   |-- test.txt
   |   |-- train.txt
   |   └── val.txt
   |-- SceneMeshes/
   |   |-- fg_actors/
   |   |-- fg_actor_occ/
   |   |-- TownXX_Opt/
   |   |   |-- bg_actors/
   |   |   └── bg_actor_occ/
   |-- TownXX_Opt_SeqXX/
   |   |-- poses/
   |   |   |-- cam_00.txt
   |   |   └── lidar.txt
   |   |-- rgb/
   |   |   |-- image_00/
   |   |   |   |-- 0000.png
   |   |   |   |-- ...
   |   |   |   └── 0999.png
   |   |   |-- ...
   |   |   |-- image_05/
   |   |   └── image_bev/
   |   |-- depth/
   |   |-- normal/
   |   |-- semantics/
   |   |-- lidar/
   |   |-- semantic_lidar/
   |   |-- occupancy/
   |   |   |-- vs_0_1/
   |   |   |-- vs_0_2_forward_view/
   |   |   └── vs_0_4_surround_view/
   |   └── traffic_info/
   |-- ...
   └── Town10HD_Opt_Seq12/


Dataset Visualization
---------------------

``vis_dataset.py`` provides a command-line interface for visualizing any data modality for a given sequence and frame. It writes PNG outputs to ``--save_dir`` (default: ``outputs/vis/``) or displays interactively with ``--interactive``.

.. code-block:: bash

    # Surround-view RGB images (6-camera grid)
    python vis_dataset.py --dataset_dir data/CarlaOccV1_mini \
        --towns Town01 --seqs Seq07 --frame_id 0 --vis_modality rgb

    # Depth maps (plasma colormap, 0–80 m)
    python vis_dataset.py --dataset_dir data/CarlaOccV1_mini \
        --towns Town01 --seqs Seq07 --frame_id 0 --vis_modality depth

    # Panoptic occupancy (Open3D 3-D viewer)
    python vis_dataset.py --dataset_dir data/CarlaOccV1_mini \
        --towns Town01 --seqs Seq07 --frame_id 0 \
        --vis_modality occupancy/vs_0_4_surround_view

    # All modalities, all frames in the sequence
    python vis_dataset.py --dataset_dir data/CarlaOccV1_mini \
        --towns Town01 --seqs Seq07 --frame_id all --vis_modality all

    # Display interactively instead of saving
    python vis_dataset.py --dataset_dir data/CarlaOccV1_mini \
        --towns Town01 --seqs Seq07 --frame_id 0 --interactive

.. list-table:: Available modalities
   :header-rows: 1
   :widths: 35 65

   * - ``--vis_modality``
     - Description
   * - ``rgb``
     - 6-camera surround RGB images (3×2 grid)
   * - ``depth``
     - 16-bit depth maps rendered as plasma colourmap (0–80 m)
   * - ``normal``
     - Surface normal images (3×2 grid)
   * - ``semantics``
     - Colour-coded semantic segmentation images
   * - ``lidar``
     - LiDAR point cloud coloured by intensity
   * - ``semantic_lidar``
     - Semantic LiDAR point cloud coloured by class
   * - ``occupancy/vs_0_1``
     - Panoptic occupancy at 0.1 m voxel resolution
   * - ``occupancy/vs_0_2_forward_view``
     - Panoptic occupancy at 0.2 m, forward-view frustum
   * - ``occupancy/vs_0_4_surround_view``
     - Panoptic occupancy at 0.4 m, surround-view volume
   * - ``all``
     - All modalities above in sequence


Coordinate system definition
----------------------------
There are four coordinate systems used in this project. To avoid ambiguity, we specify:

- `Carla/UE Coordinates <https://carla.readthedocs.io/en/latest/coordinates/#coordinates-and-transformations>`_ (left-handed):

  - X: forward, Y: right, Z: upward
  - Pitch: rotation along Y-axis, with direction X --> Z
  - Yaw: rotation along Z-axis, with direction X --> Y
  - Roll: rotation along X-axis, with direction Z --> Y

- World/Sensor/Occupancy Coordinates (right-handed):

  - X: forward, Y: left, Z: upward

- Mesh/Object Coordinates (right-handed):
  
  - X: forward, Y: upward, Z: right

- Open3d Camera Coordinates (right-handed):

  - X: right, Y: down, Z: forward

.. note::

    For ease of use, we have transformed all coordinates in CarlaOcc to right-handed coordinates.
    See `transforms.py <https://github.com/fengyi233/carlaocc/blob/main/utils/transforms.py>`_ for more details.

