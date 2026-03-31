Occupancy Generation
====================

Overview
--------

The ``occupancy_generation/`` module is used for processing the raw collected data collected from CARLA to generate high-quality derived modalities. This includes rectifying sensor artifacts to produce refined semantic depth maps, generating surface normals, and constructing dense and resampled panoptic occupancy grids. 

Running
-------

We provide a set of bash scripts to automate the generation and visualization pipelines. You can run these scripts directly from the repository root.

Step 1: Modality Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To generate all modalities (refined semantic depth, normals, and panoptic occupancy at 0.1m resolution) once for all, run:

.. code-block:: bash

   bash occupancy_generation/gen_modalities.sh

You can also generate required modalities by separately running the following commands:

.. code-block:: bash
  
  # Semantic Depth Generation
  python occupancy_generation/generators/gen_sem_depth.py
  # Normal Map Generation
  python occupancy_generation/generators/gen_normal.py
  # Panoptic Occupancy Generation
  python occupancy_generation/generators/gen_pano_occ.py



Step 2: Occupancy Resampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The raw occupancy grids (``vs_0_1``) can be resampled into different spatial extents and resolutions. Use following command to resample the occupancy grids for forward-view or surround-view tasks:

.. code-block:: bash

   bash occupancy_generation/resample_occ.sh

You can also customize your required occ GT format 



This script uses ``generators/resample_occ.py`` to downsample the occupancy into predefined configurations like ``vs_0_2_forward_view`` and ``vs_0_4_surround_view``.


Step 3: Visualization
~~~~~~~~~~~~~~~~~~~~~

To visualize the generated scene meshes and occupancy grids, run:

.. code-block:: bash

   bash occupancy_generation/vis_modalities.sh



Occupancy Resolution Settings
-----------------------------

Definitions
~~~~~~~~~~~~

- **Voxel Size** (m): Physical size of each voxel along the (x, y, z) axes.
- **Voxel Origin** (m): The coordinate of the minimum corner of the voxel grid in LiDAR Coordinate System.
- **Voxel Range** (m): The spatial coverage of the voxel grid along each axis in LiDAR Coordinate System.
- **Volume Size**: The resolution of the occupancy grid.

Configuration Table
~~~~~~~~~~~~~~~~~~~

.. list-table:: Occupancy grid configurations under different resolutions
   :header-rows: 1
   :align: center
   :widths: 20 20 20 30 20

   * - Config Type
     - Voxel Size
     - Voxel Origin
     - Voxel Range
     - Volume Size
   * - vs_0_1 (Raw)
     - 0.1
     - [-25.6, -25.6, -3]
     - [[-25.6, 51.2], [-25.6, 25.6], [-3, 10]]
     - [768, 512, 130]
   * - vs_0_2_forward_view
     - 0.2
     - [0, -25.6, -2.4]
     - [[0, 51.2], [-25.6, 25.6], [-2.4, 4]]
     - [256, 256, 32]
   * - vs_0_4_surround_view
     - 0.4
     - [-25.6, -25.6, -2.4]
     - [[-25.6, 25.6], [-25.6, 25.6], [-2.4, 4]]
     - [128, 128, 16]


Notes
~~~~~

- The **Volume Size** can be computed as the spatial extent divided by the voxel size.

- The configuration **vs_0_2_forward_view** follows the spatial setup of `KITTI-360-SSCBench <https://github.com/ai4ce/SSCBench>`_ benchmark,
  focusing on the single-view occupancy prediction task. The configuration **vs_0_4_surround_view**
  follows the spatial setup of `Occ3D-nuScenes <https://github.com/Tsinghua-MARS-Lab/Occ3D>`_ benchmark, focusing on the surround-view occupancy prediction task.
  Please note that the Voxel Range is slightly different with these two benchmarks.

- To generate your customized occupancy ground truth with a different voxel size / voxel origin / volume size,
  run the following command:

  .. code-block:: bash

    cd data_collection
    # e.g., to produce vs_0_2_forward_view
    python occ_downsample.py \
      --dataset_dir ../data/CarlaOccV1 \
      --voxel_size 0.2 \
      --save_dir_name vs_0_2_forward_view \
      --voxel_origin 0 -25.6 -2.4 \
      --volume_size 256 256 32



Panoptic Occupancy Labels
--------------------------

Label Format
~~~~~~~~~~~~

To reduce spatial redundancy, we adopt a sparse format for the occupancy labels and use the ``.npz`` format to store them. The ground truth labels contain the following fields:

- **occupancy**: Sparse occupancy labels in shape of [N, 4], where each row is [x, y, z, encoded_label] representing the voxel indices and its label.
- **voxel_size**: See above.
- **voxel_origin**: See above. 
- **volume_size**: See above. It is used to decode the occupancy labels back to the dense format.

Label Encoding and Decoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To unify the storage format of the occupancy labels, we encode the semantic labels and instance labels into a single combined label of ``uint16``. The encoding and decoding are defined as follows:

.. code-block:: python

    encoded_label = semantic_label * 1000 + instance_label % 1000
    semantic_label, instance_label = encoded_label // 1000, encoded_label % 1000

Notes
~~~~~

- For non-instance classes (i.e., stuff classes), the instance label is set to 0.
- The semantic classes should be numbered in range of [0, 64].
- The instance labels should be numbered in range of [0, 999]. Instance labels should be unique within each semantic class, and can be reused in different scenes.
