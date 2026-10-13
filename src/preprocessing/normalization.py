import numpy as np


RGB_SCALE             = 1 / 10_000  # standard for sentinel-2
GAMMA                 = 1 / 2.2


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    reflectance = np.clip(rgb.astype(np.float32) * RGB_SCALE, 0, 1)
    return reflectance ** GAMMA
