"""
Transformations scripts for diverse coordinates.

Coordinate Definition:
    Mesh Coordinates (right-handed):
        X: forward, Y: upward, Z: right
        NOTE: All meshes are exported from UE5 editor.

    Carla/UE Coordinates (left-handed):
        X: forward, Y: right, Z: upward
        Pitch: rotation along Y-axis, with direction X --> Z
        Yaw: rotation along Z-axis, with direction X --> Y
        Roll: rotation along X-axis, with direction Z --> Y
        See https://carla.readthedocs.io/en/latest/python_api/#carlarotation for more details.
        NOTE: All coordinates are in left-handed coordinate system in CARLA, 
        including sensor coordinates and carla world coordinates.

    World/Sensor/Occupancy Coordinates (right-handed):
        X: forward, Y: left, Z: upward
        NOTE: This is the uniform coordinate system we used in CarlaOcc.

    Open3D Camera Coordinates:
        X: right, Y: down, Z: forward

"""

from typing import Union, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

# ==================================================================
# Transformation Matrices
# ==================================================================
transform_mat_obj_to_carla = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

transform_mat_carla_to_world = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

transform_mat_obj_to_world = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

transform_mat_world_to_o3d = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
])


# ==================================================================
# CARLA Transformations
# ==================================================================

def carla_to_np(carla_obj, decimal=6) -> np.ndarray:
    """
    Convert CARLA objects to numpy arrays.
    
    Args:
        carla_obj: carla.Location, carla.Rotation, carla.Vector3D, or carla.Vector2D
        decimal: number of decimal places for rounding
        
    Returns:
        numpy array with the object's components
    """
    import carla

    # Define conversion mappings
    conversions = {
        carla.Location: lambda obj: [obj.x, obj.y, obj.z],
        carla.Rotation: lambda obj: [obj.pitch, obj.yaw, obj.roll],
        carla.Vector3D: lambda obj: [obj.x, obj.y, obj.z],
        carla.Vector2D: lambda obj: [obj.x, obj.y]
    }

    # Find appropriate conversion
    for carla_type, converter in conversions.items():
        if isinstance(carla_obj, carla_type):
            return np.array([round(x, decimal) for x in converter(carla_obj)])

    raise ValueError(f"Unsupported CARLA type: {type(carla_obj)}")


def carla_matrix_to_transform(matrix: np.ndarray) -> dict:
    """
    Convert a 4x4 transformation matrix from CARLA's left-handed coordinate system to 
    carla transform dictionary containing location and rotation.
    
    Args:
        matrix: 4x4 numpy array (left-handed coordinate system)
        
    Returns:    
        dict: {'location': [x, y, z], 'rotation': [pitch, yaw, roll]}
    """
    assert matrix.shape == (4, 4), "Input matrix must be 4x4"
    location = matrix[:3, 3].tolist()
    rotation = R.from_matrix(matrix[:3, :3]).as_euler('zyx', degrees=True)
    rotation = [float(rotation[1]), float(rotation[0]), float(rotation[2])]
    return {'location': location, 'rotation': rotation}


def carla_transform_to_mat(carla_transform) -> np.ndarray:
    """
    Convert CARLA transform to 4x4 transformation matrix (right-handed).
    Carla rotation order: roll --> pitch --> yaw.

    Returns:
        4x4 numpy array: transformation matrix in left-handed coordinate system
    """
    import carla
    assert isinstance(carla_transform, carla.Transform), "Input must be a carla.Transform object"

    location = vector_left_to_right(carla_transform.location)

    # right-handed conversion
    pitch = -carla_transform.rotation.pitch
    yaw = -carla_transform.rotation.yaw
    roll = carla_transform.rotation.roll

    matrix = np.eye(4)
    matrix[:3, 3] = location
    matrix[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()

    matrix = np.array(matrix, dtype=np.float32)
    return matrix


def mat_left_to_right(mat: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 transformation matrix from left-handed to right-handed.

    Args:
        mat: 4x4 numpy array (left-handed coordinate system)

    Returns:
        4x4 numpy array (right-handed coordinate system)

    """
    if mat.shape != (4, 4):
        raise ValueError("Input matrix must be 4x4")

    # Conversion matrix: flip Y-axis
    convert = np.diag([1, -1, 1, 1])
    return convert @ mat @ convert


def vector_left_to_right(vector: Union[np.ndarray, List, Tuple]) -> Union[np.ndarray, List, Tuple]:
    """
    Convert a vector from left-handed to right-handed.

    Args:
        vector: 3D numpy array, list, or tuple (left-handed coordinate system)

    Returns:
        3D numpy array, list, or tuple (right-handed coordinate system)

    """
    import carla
    if isinstance(vector, (carla.Location, carla.Vector3D)):
        vector = carla_to_np(vector)

    assert len(vector) == 3, "Input vector must be 3D"

    if isinstance(vector, np.ndarray):
        return vector * np.array([1, -1, 1])
    elif isinstance(vector, list):
        return [vector[0], -vector[1], vector[2]]
    elif isinstance(vector, tuple):
        return (vector[0], -vector[1], vector[2])
    else:
        raise ValueError("Input vector must be numpy array, list, or tuple")


def dict_to_carla_transform(transform_dict: dict):
    import carla
    location = carla.Location(**transform_dict['location'])
    rotation = carla.Rotation(**transform_dict['rotation'])
    return carla.Transform(location, rotation)
