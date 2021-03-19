from dataclasses import dataclass
import numpy as np
from helper_functions import skew
from typing import List


@dataclass
class GeneratedData:
    y_s: np.ndarray
    y_w: np.ndarray
    Hs: np.ndarray
    h_s: np.ndarray
    r: np.ndarray
    is_saturated: bool
    num_saturated_w: int
    saturated_indices: List[int] = None
    average_w: np.ndarray = None

    def __post_init__(self):
        # Saturates at 2000
        self.y = np.concatenate([self.y_s, self.y_w], axis=0)
        self.h_w = self.y_w
        self.h_of_w = np.concatenate([self.h_s, self.h_w], axis=0)
        self.H = np.concatenate([self.Hs, np.zeros((len(self.y_w), 6))], axis=0)

    def generate_h_w(self, w: np.ndarray) -> np.ndarray:
        full_h_w = np.concatenate([w] * Params.num_w, axis=0)
        cropped_h_w = np.delete(full_h_w, self.saturated_indices, axis=0)
        return cropped_h_w
        # return np.concatenate([w] * Params.num_w, axis=0)

    def generate_hs_w(self, w: np.ndarray) -> np.ndarray:
        omega_w_2 = skew(w) @ skew(w)
        return np.concatenate([omega_w_2 @ r_i.reshape(3, 1) for r_i in self.r], axis=0)

    def h(self, w: np.ndarray) -> np.ndarray:
        return np.concatenate([self.generate_hs_w(w), self.generate_h_w(w)], axis=0)





class Params:
    # System Parameters
    """
    System Paramter Constants. Set these values before running simulation!
    """
    num_w = 4
    num_s = 4
    r = np.array([1, 0, 0])
