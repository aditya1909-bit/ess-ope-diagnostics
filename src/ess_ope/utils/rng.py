from __future__ import annotations

import random
from typing import Optional

import numpy as np


def seed_all(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
