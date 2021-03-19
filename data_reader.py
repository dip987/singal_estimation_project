import math

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from cycler import cycler
import glob
from itertools import cycle


def load_data(file_path: Path) -> np.ndarray:
    all_data = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=4).replace(to_replace='+nan',
                                                                                              value=np.NaN).fillna(
        method='pad').to_numpy()
    all_data = np.array(all_data[:, [1, 2, 3, 4, 5, 6, 7, 14, 15, 16]], dtype=float)  # Crop the unnecessary data
    all_data = all_data[2001:2301, :]  # Keeping fewer rows for faster calculations
    accelerometer_scale_factor = 9.80665
    all_data[:, :3] *= accelerometer_scale_factor
    all_data[:, -3:] *= accelerometer_scale_factor
    return all_data


Ts = 1 / 250
data_paths = glob.glob(r'C:\Users\sadip\Documents\winter_21\Signal estimation\Project\openshoe-omi\*\data_inert.txt',
                       recursive=True)
# Read data
files = []
for path in data_paths:
    files.append(load_data(path))

marker = cycle(
    ('.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'))

plt.close('all')
default_cycler = (cycler(color=['r', 'g', 'b']))
plt.rc('axes', prop_cycle=default_cycler)
plt.rc('lines', linewidth=1)
plt.figure()
for file in files:
    plt.plot(file[:, 3:6], marker=next(marker))
