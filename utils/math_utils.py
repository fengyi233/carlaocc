"""
Math utility functions for geometric operations.
"""
from typing import List, Union

import numpy as np


def bbox_intersects(
        bbox1_min: Union[List[float], np.ndarray],
        bbox1_max: Union[List[float], np.ndarray],
        bbox2_min: Union[List[float], np.ndarray],
        bbox2_max: Union[List[float], np.ndarray]
) -> bool:
    """
    Check if two 3D axis-aligned bounding boxes intersect.
    
    Two bounding boxes intersect if they overlap in all three dimensions (X, Y, Z).
    
    Args:
        bbox1_min: Minimum corner [x, y, z] of the first bounding box
        bbox1_max: Maximum corner [x, y, z] of the first bounding box
        bbox2_min: Minimum corner [x, y, z] of the second bounding box
        bbox2_max: Maximum corner [x, y, z] of the second bounding box
        
    Returns:
        True if the bounding boxes intersect, False otherwise
        
    Examples:
        >>> bbox_intersects([0, 0, 0], [10, 10, 10], [5, 5, 5], [15, 15, 15])
        True
        >>> bbox_intersects([0, 0, 0], [10, 10, 10], [20, 20, 20], [30, 30, 30])
        False
    """
    bbox1_min = np.asarray(bbox1_min)
    bbox1_max = np.asarray(bbox1_max)
    bbox2_min = np.asarray(bbox2_min)
    bbox2_max = np.asarray(bbox2_max)

    # Two boxes intersect if they overlap in all three dimensions
    # They overlap in a dimension if: bbox1_max >= bbox2_min AND bbox1_min <= bbox2_max
    return np.all(bbox1_max >= bbox2_min) and np.all(bbox1_min <= bbox2_max)


def bbox_contains_point(
        bbox_min: Union[List[float], np.ndarray],
        bbox_max: Union[List[float], np.ndarray],
        point: Union[List[float], np.ndarray]
) -> bool:
    """
    Check if a point is inside a 3D axis-aligned bounding box.
    
    Args:
        bbox_min: Minimum corner [x, y, z] of the bounding box
        bbox_max: Maximum corner [x, y, z] of the bounding box
        point: Point coordinates [x, y, z]
        
    Returns:
        True if the point is inside the bounding box, False otherwise
        
    Examples:
        >>> bbox_contains_point([0, 0, 0], [10, 10, 10], [5, 5, 5])
        True
        >>> bbox_contains_point([0, 0, 0], [10, 10, 10], [15, 15, 15])
        False
    """
    bbox_min = np.asarray(bbox_min)
    bbox_max = np.asarray(bbox_max)
    point = np.asarray(point)

    return np.all(point >= bbox_min) and np.all(point <= bbox_max)


def bbox_volume(
        bbox_min: Union[List[float], np.ndarray],
        bbox_max: Union[List[float], np.ndarray]
) -> float:
    """
    Calculate the volume of a 3D axis-aligned bounding box.
    
    Args:
        bbox_min: Minimum corner [x, y, z] of the bounding box
        bbox_max: Maximum corner [x, y, z] of the bounding box
        
    Returns:
        Volume of the bounding box
        
    Examples:
        >>> bbox_volume([0, 0, 0], [10, 10, 10])
        1000.0
    """
    bbox_min = np.asarray(bbox_min)
    bbox_max = np.asarray(bbox_max)
    extents = bbox_max - bbox_min
    return np.prod(extents)


def bbox_intersection_volume(
        bbox1_min: Union[List[float], np.ndarray],
        bbox1_max: Union[List[float], np.ndarray],
        bbox2_min: Union[List[float], np.ndarray],
        bbox2_max: Union[List[float], np.ndarray]
) -> float:
    """
    Calculate the volume of intersection between two 3D axis-aligned bounding boxes.
    
    Args:
        bbox1_min: Minimum corner [x, y, z] of the first bounding box
        bbox1_max: Maximum corner [x, y, z] of the first bounding box
        bbox2_min: Minimum corner [x, y, z] of the second bounding box
        bbox2_max: Maximum corner [x, y, z] of the second bounding box
        
    Returns:
        Volume of intersection (0 if boxes don't intersect)
        
    Examples:
        >>> bbox_intersection_volume([0, 0, 0], [10, 10, 10], [5, 5, 5], [15, 15, 15])
        125.0
    """
    if not bbox_intersects(bbox1_min, bbox1_max, bbox2_min, bbox2_max):
        return 0.0

    bbox1_min = np.asarray(bbox1_min)
    bbox1_max = np.asarray(bbox1_max)
    bbox2_min = np.asarray(bbox2_min)
    bbox2_max = np.asarray(bbox2_max)

    # Calculate intersection box
    intersection_min = np.maximum(bbox1_min, bbox2_min)
    intersection_max = np.minimum(bbox1_max, bbox2_max)

    return bbox_volume(intersection_min, intersection_max)
