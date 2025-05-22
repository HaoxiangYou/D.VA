import torch
import copy

def safe_deepcopy(obj):
    if isinstance(obj, torch.Tensor):
        clone = obj.detach().clone()
        clone.requires_grad_(obj.requires_grad)
        return clone

    elif isinstance(obj, dict):
        return {k: safe_deepcopy(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [safe_deepcopy(v) for v in obj]

    elif isinstance(obj, tuple):
        return tuple(safe_deepcopy(v) for v in obj)

    elif isinstance(obj, set):
        return {safe_deepcopy(v) for v in obj}

    elif hasattr(obj, '__dict__'):
        # For custom class instances, create a shallow instance and deepcopy its attrs
        new_obj = obj.__class__.__new__(obj.__class__)
        for k, v in obj.__dict__.items():
            setattr(new_obj, k, safe_deepcopy(v))
        return new_obj

    else:
        try:
            return copy.deepcopy(obj)
        except Exception as e:
            print(f"Warning: fallback shallow copy for {obj}: {e}")
            return obj