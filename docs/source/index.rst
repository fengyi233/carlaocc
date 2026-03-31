.. image:: _static/carlaocc.png
   :alt: CarlaOcc Logo
   :align: center
   :width: 60%
   :target: https://github.com/fengyi233/CarlaOcc

Welcome to CarlaOcc Dataset Documentation!
==========================================

.. raw:: html

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
   <a href="https://github.com/fengyi233/CarlaOcc" style="text-decoration:none;">
      <strong><img src="https://img.icons8.com/material-outlined/48/github.png" alt="GitHub" width="20" height="20" style="vertical-align:middle; margin-right:4px;"/>GitHub</strong>
   </a>
   </p>
   <p align="center">
     <img src="_static/town10.gif" style="max-width: 100%; height: auto;" alt="CarlaOcc demo" />
   </p>
   <br/>

**CarlaOcc** is a high-fidelity and physically-consistent dataset created from the CARLA simulator.
To facilitate the development of 3D perception, scene understanding, and occupancy prediction tasks, 
CarlaOcc provides comprehensive multi-modal data and fine-grained annotations, including:

- Surround-view/Stereo RGB images
- Panoptic/semantic occupancy ground truth
- Depth images
- Surface normal images
- Semantic segmentation images
- LiDAR/Semantic LiDAR point clouds
- Traffic metadata (e.g., 3D bounding boxes, vehicle velocities)
- Camera/LiDAR poses
- Static meshes of the scene
- Background scene layouts

This project provides complete functionality for data collection, scene exportation, and occupancy ground truth generation.

Getting Started
---------------

We recommend the readers to explore this project by the following sequences:

- Dataset Tutorial (`CarlaOcc/tutorials/`):
    Have a quick look of our dataset
- Data Collection (`CarlaOcc/data_collection/`):
    Collect sensor data from the CARLA simulator
- Scene Exportation (`CarlaOcc/scene_exportation/`):
    Export scene geometry layout and semantic information
- Occupancy Grid Generation (`CarlaOcc/occupancy_generation/`):
    Generate dense and sparse occupancy grid representations


Documentation
-------------

.. toctree::
   :maxdepth: 2

   dataset_tutorial
   data_collection
   scene_exportation
   occ_generation


.. note::

   This project is under active development. Feel free to contribute to the project by opening an issue or a pull request.