"""
Semantic label definitions and mappings for CarlaOcc and NuScenes.
"""
from collections import namedtuple

import numpy as np

# ========================================================================================
# CarlaOcc Semantic Labels (30 classes, CarlaUE5)
# ========================================================================================
# Label namedtuple: Defines semantic label structure for CarlaOcc dataset
# Fields:
#   - name: str          - Human-readable label name (e.g., "Car", "Road")
#   - id: int            - CarlaOcc label ID (0-29)
#   - is_instance: bool  - Whether this label represents instance-level objects (True) or stuff (False)
#   - color: tuple       - RGB color for visualization (0-255 range)
#   - nuscenes_id: int   - Mapped NuScenes Occ3D label ID (0-17)
#   - kitti360_id: int   - Mapped KITTI-360 SSCBench label ID (0-18)
Label = namedtuple("Label",
                   ["name", "id", "is_instance", "color", "nuscenes_id", "kitti360_id"])

labels = [
    Label(name="None"        , id=0 , is_instance=False, color=(0, 0, 0)      , nuscenes_id=17, kitti360_id=0 ),
    Label(name="Road"        , id=1 , is_instance=False, color=(128, 64, 128) , nuscenes_id=11, kitti360_id=7 ),
    Label(name="Sidewalk"    , id=2 , is_instance=False, color=(244, 35, 232) , nuscenes_id=13, kitti360_id=9 ),
    Label(name="Building"    , id=3 , is_instance=False, color=(70, 70, 70)   , nuscenes_id=15, kitti360_id=11),
    Label(name="Wall"        , id=4 , is_instance=False, color=(102, 102, 156), nuscenes_id=15, kitti360_id=17),
    Label(name="Fence"       , id=5 , is_instance=False, color=(190, 153, 153), nuscenes_id=1 , kitti360_id=12),
    Label(name="Pole"        , id=6 , is_instance=True , color=(153, 153, 153), nuscenes_id=15, kitti360_id=15),
    Label(name="TrafficLight", id=7 , is_instance=True , color=(250, 170, 30) , nuscenes_id=0 , kitti360_id=16),
    Label(name="TrafficSign" , id=8 , is_instance=True , color=(220, 220, 0)  , nuscenes_id=0 , kitti360_id=16),
    Label(name="Vegetation"  , id=9 , is_instance=False, color=(107, 142, 35) , nuscenes_id=16, kitti360_id=13),
    Label(name="Terrain"     , id=10, is_instance=False, color=(152, 251, 152), nuscenes_id=14, kitti360_id=14),
    Label(name="Sky"         , id=11, is_instance=False, color=(70, 130, 180) , nuscenes_id=17, kitti360_id=0 ),
    Label(name="Pedestrian"  , id=12, is_instance=True , color=(220, 20, 60)  , nuscenes_id=7 , kitti360_id=6 ),
    Label(name="Rider"       , id=13, is_instance=True , color=(255, 0, 0)    , nuscenes_id=7 , kitti360_id=6 ),
    Label(name="Car"         , id=14, is_instance=True , color=(0, 0, 142)    , nuscenes_id=4 , kitti360_id=1 ),
    Label(name="Truck"       , id=15, is_instance=True , color=(0, 0, 70)     , nuscenes_id=10, kitti360_id=4 ),
    Label(name="Bus"         , id=16, is_instance=True , color=(0, 60, 100)   , nuscenes_id=3 , kitti360_id=5 ),
    Label(name="Train"       , id=17, is_instance=True , color=(0, 80, 100)   , nuscenes_id=9 , kitti360_id=5 ),
    Label(name="Motorcycle"  , id=18, is_instance=True , color=(0, 0, 230)    , nuscenes_id=6 , kitti360_id=3 ),
    Label(name="Bicycle"     , id=19, is_instance=True , color=(119, 11, 32)  , nuscenes_id=2 , kitti360_id=2 ),
    Label(name="Static"      , id=20, is_instance=False, color=(110, 190, 160), nuscenes_id=0 , kitti360_id=18),
    Label(name="Dynamic"     , id=21, is_instance=False, color=(170, 120, 50) , nuscenes_id=0 , kitti360_id=18),
    Label(name="Other"       , id=22, is_instance=False, color=(55, 90, 80)   , nuscenes_id=0 , kitti360_id=18),
    Label(name="Water"       , id=23, is_instance=False, color=(45, 60, 150)  , nuscenes_id=12, kitti360_id=10),
    Label(name="RoadLine"    , id=24, is_instance=False, color=(157, 234, 50) , nuscenes_id=11, kitti360_id=7 ),
    Label(name="Ground"      , id=25, is_instance=False, color=(81, 0, 81)    , nuscenes_id=14, kitti360_id=10),
    Label(name="Bridge"      , id=26, is_instance=False, color=(150, 100, 100), nuscenes_id=15, kitti360_id=17),
    Label(name="RailTrack"   , id=27, is_instance=False, color=(230, 150, 140), nuscenes_id=12, kitti360_id=10),
    Label(name="GuardRail"   , id=28, is_instance=False, color=(180, 165, 180), nuscenes_id=1 , kitti360_id=12),
    Label(name="Rock"        , id=29, is_instance=False, color=(180, 130, 70) , nuscenes_id=14, kitti360_id=14),
]

# ========================================================================================
# NuScenes Semantic Labels (18 classes, Occ3d)
# ========================================================================================
NusLabel = namedtuple("NusLabel", ["name", "id", "is_instance", "color"])

nuscenes_labels = [
    NusLabel("others",              0,  False, (0, 0, 0)),
    NusLabel("barrier",             1,  False, (255, 120, 50)),
    NusLabel("bicycle",             2,  True,  (255, 192, 203)),
    NusLabel("bus",                 3,  True,  (255, 255, 0)),
    NusLabel("car",                 4,  True,  (0, 150, 245)),
    NusLabel("construction_vehicle", 5, True,  (0, 255, 255)),
    NusLabel("motorcycle",          6,  True,  (200, 180, 0)),
    NusLabel("pedestrian",          7,  True,  (255, 0, 0)),
    NusLabel("traffic_cone",        8,  False, (255, 240, 150)),
    NusLabel("trailer",             9,  True,  (135, 60, 0)),
    NusLabel("truck",              10,  True,  (160, 32, 240)),
    NusLabel("driveable_surface",  11,  False, (255, 0, 255)),
    NusLabel("other_flat",         12,  False, (175, 0, 75)),
    NusLabel("sidewalk",           13,  False, (75, 0, 75)),
    NusLabel("terrain",            14,  False, (150, 240, 80)),
    NusLabel("manmade",            15,  False, (230, 230, 250)),
    NusLabel("vegetation",         16,  False, (0, 175, 0)),
    NusLabel("free",               17,  False, (255, 255, 255)),
]

nuscenes_id2label = {label.id: label for label in nuscenes_labels}
nuscenes_name2label = {label.name: label for label in nuscenes_labels}
nuscenes_id2color = {label.id: label.color for label in nuscenes_labels}

# ========================================================================================
# KITTI-360 Semantic Labels (19 classes, SSCBench)
# ========================================================================================
SSCLabel = namedtuple("SSCLabel", ["name", "id", "is_instance", "color"])

kitti360_labels = [
    SSCLabel("unlabeled",       0,  False, (0, 0, 0)),
    SSCLabel("car",             1,  True,  (100, 150, 245)),
    SSCLabel("bicycle",         2,  True,  (100, 230, 245)),
    SSCLabel("motorcycle",      3,  True,  (30, 60, 150)),
    SSCLabel("truck",           4,  True,  (80, 30, 180)),
    SSCLabel("other-vehicle",   5,  True,  (0, 0, 255)),
    SSCLabel("person",          6,  True,  (255, 30, 30)),
    SSCLabel("road",            7,  False, (255, 0, 255)),
    SSCLabel("parking",         8,  False, (255, 150, 255)),
    SSCLabel("sidewalk",        9,  False, (75, 0, 75)),
    SSCLabel("other-ground",    10, False, (175, 0, 75)),
    SSCLabel("building",        11, False, (255, 200, 0)),
    SSCLabel("fence",           12, False, (255, 120, 50)),
    SSCLabel("vegetation",      13, False, (0, 175, 0)),
    SSCLabel("terrain",         14, False, (150, 240, 80)),
    SSCLabel("pole",            15, False, (255, 240, 150)),
    SSCLabel("traffic-sign",    16, False, (255, 0, 0)),
    SSCLabel("other-structure", 17, False, (255, 150, 0)),
    SSCLabel("other-object",    18, False, (50, 255, 255)),
]

kitti360_id2label = {label.id: label for label in kitti360_labels}
kitti360_name2label = {label.name: label for label in kitti360_labels}
kitti360_id2color = {label.id: label.color for label in kitti360_labels}

# Basic Lookups
name2label = {label.name: label for label in labels}
id2label = {label.id: label for label in labels}
id2color = {label.id: label.color for label in labels}
color2id = {label.color: label.id for label in labels}
color_LUT = np.array([np.array(label.color) / 255.0 for label in labels])

# CarlaOcc to NuScenes Mappings
carla_id_to_nuscenes_id = {label.id: label.nuscenes_id for label in labels}
# Build color mapping through nuscenes_id
carla_id_to_nuscenes_color = {
    label.id: nuscenes_id2color.get(label.nuscenes_id, (0, 0, 0))
    for label in labels
}

# CarlaOcc to KITTI-360 SSCBench Mappings
carla_id_to_kitti360_id = {label.id: label.kitti360_id for label in labels}
# Build color mapping through kitti360_id
carla_id_to_kitti360_color = {
    label.id: kitti360_id2color.get(label.kitti360_id, (0, 0, 0))
    for label in labels
}

# ========================================================================================
# Z-order for Rendering Priority (lower = rendered first, higher = rendered last)
# Used when filling occupancy grids to control layering
# ========================================================================================
sem_z_order = {
    # Ground layers: lowest priority
    0:  0,    # None
    23: 1,   # Water
    10: 2,   # Terrain
    25: 3,   # Ground
    1:  4,    # Road
    24: 5,   # RoadLine
    2:  6,    # Sidewalk
    27: 7,   # RailTrack
    29: 8,   # Rock
    # Vertical structures
    5:  9,    # Fence
    28: 10,  # GuardRail
    4:  11,   # Wall
    3:  12,   # Building
    26: 13,  # Bridge
    6:  14,   # Pole
    7:  15,   # TrafficLight
    8:  16,   # TrafficSign
    # Static and dynamic objects
    20: 17,  # Static
    21: 18,  # Dynamic
    22: 19,  # Other
    9:  20,   # Vegetation
    # Movable agents: highest priority
    12: 21,  # Pedestrian
    13: 22,  # Rider
    19: 23,  # Bicycle
    18: 24,  # Motorcycle
    14: 25,  # Car
    15: 26,  # Truck
    16: 27,  # Bus
    17: 28,  # Train
    # Sky
    11: 29,  # Sky
}
lut_sem_prior = np.array([sem_z_order[label.id] for label in labels], dtype=np.int32)


# ========================================================================================
# Conversion Functions
# ========================================================================================
def convert_carla_to_nuscenes(carla_occupancy: np.ndarray) -> np.ndarray:
    """
    Convert CarlaOcc occupancy labels to NuScenes labels.
    
    Args:
        carla_occupancy: Array with CarlaOcc label IDs (shape: [N,] or [N, 4] sparse format)
    
    Returns:
        nuscenes_occupancy: Array with NuScenes label IDs (same shape as input)
    """
    if carla_occupancy.ndim == 1:
        # Dense array or 1D label array
        return np.vectorize(carla_id_to_nuscenes_id.get)(carla_occupancy, 0)
    elif carla_occupancy.ndim == 2 and carla_occupancy.shape[1] == 4:
        # Sparse format [x, y, z, label_id]
        result = carla_occupancy.copy()
        result[:, 3] = np.vectorize(carla_id_to_nuscenes_id.get)(carla_occupancy[:, 3], 0)
        return result
    else:
        # 3D dense grid
        return np.vectorize(carla_id_to_nuscenes_id.get)(carla_occupancy, 0)


def get_nuscenes_color_map(carla_ids: np.ndarray) -> np.ndarray:
    """
    Get NuScenes colors for CarlaOcc label IDs.
    
    Args:
        carla_ids: Array of CarlaOcc label IDs
    
    Returns:
        colors: Array of RGB colors [N, 3]
    """
    # Vectorized mapping: CarlaOcc ID -> NuScenes ID -> NuScenes Color
    colors = np.zeros((len(carla_ids), 3), dtype=np.uint8)
    for i, carla_id in enumerate(carla_ids):
        # Get NuScenes ID from CarlaOcc ID
        nuscenes_id = carla_id_to_nuscenes_id.get(carla_id, 0)
        # Get color from NuScenes ID
        colors[i] = nuscenes_id2color.get(nuscenes_id, (0, 0, 0))
    return colors


def convert_carla_to_kitti360(carla_occupancy: np.ndarray) -> np.ndarray:
    """
    Convert CarlaOcc occupancy labels to KITTI-360 SSCBench labels.
    
    Args:
        carla_occupancy: Array with CarlaOcc label IDs (shape: [N,] or [N, 4] sparse format)
    
    Returns:
        kitti360_occupancy: Array with KITTI-360 label IDs (same shape as input)
    """
    if carla_occupancy.ndim == 1:
        # Dense array or 1D label array
        return np.vectorize(carla_id_to_kitti360_id.get)(carla_occupancy, 0)
    elif carla_occupancy.ndim == 2 and carla_occupancy.shape[1] == 4:
        # Sparse format [x, y, z, label_id]
        result = carla_occupancy.copy()
        result[:, 3] = np.vectorize(carla_id_to_kitti360_id.get)(carla_occupancy[:, 3], 0)
        return result
    else:
        # 3D dense grid
        return np.vectorize(carla_id_to_kitti360_id.get)(carla_occupancy, 0)


def get_kitti360_color_map(carla_ids: np.ndarray) -> np.ndarray:
    """
    Get KITTI-360 SSCBench colors for CarlaOcc label IDs.
    
    Args:
        carla_ids: Array of CarlaOcc label IDs
    
    Returns:
        colors: Array of RGB colors [N, 3]
    """
    # Vectorized mapping: CarlaOcc ID -> KITTI-360 ID -> KITTI-360 Color
    colors = np.zeros((len(carla_ids), 3), dtype=np.uint8)
    for i, carla_id in enumerate(carla_ids):
        # Get KITTI-360 ID from CarlaOcc ID
        kitti360_id = carla_id_to_kitti360_id.get(carla_id, 0)
        # Get color from KITTI-360 ID
        colors[i] = kitti360_id2color.get(kitti360_id, (0, 0, 0))
    return colors
