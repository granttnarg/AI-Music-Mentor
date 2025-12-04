import numpy as np


def convert_numpy_to_python(obj):
    """
    Recursively convert NumPy types to Python native types for JSON serialization.

    Args:
        obj: Object potentially containing NumPy types (dict, list, ndarray, or scalar)

    Returns:
        Object with all NumPy types converted to Python native types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj
