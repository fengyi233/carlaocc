import numpy as np


def formatter(data, precision=6):
    """
    Format data for consistent precision in serialization.
    
    Args:
        data: Data to format (supports nested structures)
        precision: Number of decimal places for float values
        
    Returns:
        Formatted data with consistent precision
    """
    # Handle numpy types
    if isinstance(data, np.ndarray):
        return formatter(data.tolist(), precision)
    elif isinstance(data, np.floating):
        formatted = f"{float(data):.{precision}f}".rstrip('0').rstrip('.')
        return float(formatted)
    elif isinstance(data, np.integer):
        return int(data)

    # Handle basic types
    elif isinstance(data, float):
        formatted = f"{data:.{precision}f}".rstrip('0').rstrip('.')
        return float(formatted)
    elif isinstance(data, (int, str, bool)):
        return data

    # Handle containers
    elif isinstance(data, list):
        return [formatter(item, precision) for item in data]
    elif isinstance(data, dict):
        return {k: formatter(v, precision) for k, v in data.items()}
    elif isinstance(data, tuple):
        return tuple(formatter(item, precision) for item in data)

    # Handle None
    elif data is None:
        return None

    else:
        raise TypeError(f'Unsupported type: {type(data)}')
