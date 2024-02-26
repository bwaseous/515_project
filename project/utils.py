import os

from pathlib import Path

from typing import Union

import numpy as np
import torch
import random

def create_dir(path: Union[str, Path]) -> None:

    return

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return None