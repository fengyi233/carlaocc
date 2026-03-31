"""
CARLA utility functions for vehicle type classification and semantic mapping.
"""

# CARLA vehicle type to semantic class mapping
VEHICLE_TYPE_MAPPING = {
    "Car": [
        "vehicle.taxi.ford",
        "vehicle.nissan.patrol",
        "vehicle.dodgecop.charger",
        "vehicle.dodge.charger",
        "vehicle.lincoln.mkz",
        "vehicle.mini.cooper",
        "vehicle.ue4.chevrolet.impala",
        "vehicle.ue4.mercedes.ccc",
        "vehicle.ue4.ford.mustang",
        "vehicle.ue4.bmw.grantourer",
        "vehicle.ue4.ford.crown",
        "vehicle.ue4.audi.tt"
    ],
    "Truck": [
        "vehicle.carlacola.actors",
        "vehicle.firetruck.actors",
        "vehicle.ambulance.ford",
        "vehicle.sprinter.mercedes"
    ],
    "Bus": [
        "vehicle.fuso.mitsubishi"
    ]
}


def get_vehicle_semantic_class(carla_vehicle_type: str) -> str:
    """
    Get semantic class name from CARLA vehicle type.
    
    Args:
        carla_vehicle_type: CARLA vehicle type identifier (e.g., "vehicle.taxi.ford")
    
    Returns:
        Semantic class name: "Car", "Truck", or "Bus"
    
    Examples:
        >>> get_vehicle_semantic_class("vehicle.taxi.ford")
        'Car'
        >>> get_vehicle_semantic_class("vehicle.firetruck.actors")
        'Truck'
    """
    for semantic_class, vehicle_list in VEHICLE_TYPE_MAPPING.items():
        if carla_vehicle_type in vehicle_list:
            return semantic_class

    # Fallback: default to "Car" for unknown types
    raise ValueError(f"Unknown vehicle type: {carla_vehicle_type}")
