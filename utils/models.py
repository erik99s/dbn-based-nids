import torch

from models import (
    MLP,
    DBN,
    AE_1st,
    AE_2nd,
    AE_LSTM
)


def load_model(model_name: str, params: dict) -> torch.nn.Module:
    """Load Model."""
    if model_name == "MLP":
        return MLP(**params)
    elif model_name == "DBN":
        return DBN(**params)
    elif model_name == "AE_1st":
        return AE_1st(**params)
    elif model_name == "AE_2nd":
        return AE_2nd(**params)
    else:
        raise NotImplementedError(f"The model {model_name} is not implemented")
