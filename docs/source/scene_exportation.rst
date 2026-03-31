Scene Exportation
==================

Overview
--------

The ``Scene Exportation/`` module exports scene geometry and semantic annotations from the CARLA simulator. It consists of two main stages:

1. **Asset Exportation** (``export_scene.py``): Exports Actors from the scene to FBX format within the Unreal Engine editor and generates scene metadata JSON files.
2. **Scene Reconstruction** (``actor_recon.py``): Converts exported FBX files to OBJ format, performs mesh preprocessing, voxelization, and generates complete scene information.

Running
-------

Step 1: Asset Exportation
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Open the CARLA Unreal Engine 5 project
2. Load the map to export (e.g., Town01_Opt)
3. Run the Python script in python/cmd console:

.. code-block:: bash

    path/to/your/script/export_scene.py


Step 2: Scene Reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run ``recon_actor.py`` to post-process the exported FBX files:

.. code-block:: bash

   cd scene_exportation
   python recon_actor.py



Output Data Format
------------------

The exported meshes and reconstructed scenes will be saved as follows:

.. code-block::
   
   CarlaOccV1/
   └─ SceneMeshes/
      ├── fg_actors/                           # Foreground actors
      │   ├── Car/
      │   │   ├── vehicle_taxi_ford.fbx        # Exported vehicle mesh in FBX format
      │   │   ├── vehicle_taxi_ford.obj        # Converted vehicle mesh in OBJ format
      │   │   └── ...
      │   ├── Truck/
      │   ├── Bus/
      │   └── Pedestrians/                    
      │       ├── AS_walking04_G3.fbx          # Pedestrian animation sequence in FBX format
      │       ├── frame_info.json              # Animation frame count mapping
      │       └── standard_walking/            # Blender-exported frame-by-frame meshes
      │           ├── frame_xx.fbx
      │           └── frame_xx.obj
      │
      ├── fg_actor_occ/                        # Foreground actor voxelization results in NPZ format
      │   ├── Car/                             # Vehicle occupancy grids
      │   ├── Truck/
      │   ├── Bus/
      │   └── Pedestrians/
      │       └── standard_walking/
      │           └── frame_xx.npz
      │
      │── Town01_Opt/                          # Town-specific background actors
      │   ├── exported_scene_info.json         # Scene metadata from UE5 export
      │   ├── actor_info.json                  # Actor info
      │   ├── Town01_Opt.obj                   # Merged scene mesh (optional)
      │   ├── bg_actors/                       # Background actor OBJ files
      │   │   ├── Building_001.fbx             # Exported background actor mesh in FBX format
      │   │   └── Building_001.obj             # Converted background actor mesh in OBJ format
      │   │   ├── landscape.fbx                # Exported large actors
      │   │   ├── landscape_split_x001_y000.obj # Converted large actors split into chunks
      │   │   └── ...
      │   └── bg_actor_occ/                   # Background actor voxelization results
      │       ├── Building_001.npz
      │       ├── Actor_0_split_x023_y000.npz
      │       └── ...
      │── ...
      └── Town10HD_Opt/

