import os

from pathlib import Path

from typing import Union

import numpy as np
import torch

def create_dir(path: Union[str, Path]) -> None:

    return

def set_seed(seed: int) -> None:
    np.random.seed(seed)

    torch.seed(seed)
    torch.cuda.seed_all(seed)
    return None