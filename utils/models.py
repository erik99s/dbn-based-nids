import torch

from models import (
    MLP,
    DBN,
    AE,
)


def load_model(model_name: str, params: dict) -> torch.nn.Module:
    """Load Model."""
    if model_name == "MLP":
        return MLP(**params)
    elif model_name == "DBN":
        return DBN(**params)
    elif model_name == "AE":
        return AE(**params)
    else:
        raise NotImplementedError(f"The model {model_name} is not implemented")
