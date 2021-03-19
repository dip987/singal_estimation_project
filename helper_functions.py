import numpy as np


def to_rad(x):
    return x * np.pi / 180


def to_degree(x):
    return x * 180 / np.pi


def skew(a: np.ndarray) -> np.ndarray:
    a = a.flatten()
    assert len(a) == 3
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])


def rmse(true_w: np.ndarray, estimated_w: np.ndarray) -> float:
    return np.sqrt(np.average(np.square(true_w - estimated_w), axis=1))


def A(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return skew(u).T @ skew(v) + skew(np.cross(v.T, u.T))


def generate_phi(w_prime: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.concatenate([w_prime.T, s.T], axis=1).T


def calculate_CRLB_normal(w: np.ndarray, alpha: float, sig_s: float, sig_w: float, num_s, num_w) -> float:
    """
    Determines the Cramer-Rao Lower bound using the formula from the paper for the unsaturated case.

    :param w: true angular velocity in rad/sec
    :param alpha: distance between each of the sensors and the origin in meters
    :param sig_s: std. dev. for linear acceleration in m/s
    :param sig_w: std. dev. for angular velocity in rad/sec
    :param num_s: number of accelerometer sensors
    :param num_w: number of gyroscopes
    :return: CRLB value along each axis
    """
    fisher = num_w / sig_w ** 2 * np.identity(3) + alpha ** 2 * (num_s ** 2 - num_s) / (
            6 * sig_s ** 2) \
             * np.array([[2 * w[0] ** 2 + w[1] ** 2, w[0] * w[1], 2 * w[0] * w[2]],
                         [w[0] * w[1], 2 * w[1] ** 2 + w[0] ** 2, 2 * w[1] * w[2]],
                         [2 * w[0] * w[2], 2 * w[1] * w[2], 4 * w[2] ** 2]]).reshape(3, 3)
    CRLB = np.linalg.inv(fisher)
    return np.array([CRLB[0, 0], CRLB[1, 1], CRLB[2, 2]])


def calculate_CRLB_saturated(w: np.ndarray, alpha: float, sig_s: float, sig_w: float, num_s: float, num_w: float,
                             axis: int) -> float:
    """
    Determines the Cramer-Rao Lower bound using the formula from the paper for the saturated case.

    :param w: true angular velocity in rad/sec
    :param alpha: distance between each of the sensors and the origin in meters
    :param sig_s: std. dev. for linear acceleration in m/s
    :param sig_w: std. dev. for angular velocity in rad/sec
    :param num_s: number of accelerometer sensors
    :param num_w: number of gyroscopes
    :param axis: Along which axis to calculate. Usually some axes would give an inf value
    :return:  CRLB value along the given axis
    """
    if axis == 1 or axis == 0:
        return 6 * sig_s ** 2 / (alpha ** 2 * (num_s ** 2 - num_s)) / (w[0] ** 2 + w[1] ** 2)
    else:
        return 6 * sig_s ** 2 / (alpha ** 2 * (num_s ** 2 - num_s)) / (2 * w[2] ** 2)


def crlb_plot_wrapper(w: float, w_along: str, alpha: float, sig_s: float, sig_w: float, num_s, num_w, plot_axis: int,
                      is_saturated: bool) -> float:
    """
    plot the CRLB using a given w (as the x-axis) in degrees/sec
    
    :param w: Single value. The function will automatically create the vector from this
    :param w_along: valid values 'x', 'y', 'z' and 'all' (In case of all, it divides by root 3)
    :param alpha: distance from IMU to the origin
    :param sig_s: in meters/s
    :param sig_w: in rads/s
    :param num_s: number of accelerometers
    :param num_w: number of gyroscopes
    :param plot_axis: Returns a single CRLB value along that axis. 0,1 or 2
    :param is_saturated: Saturated or unsaturated?
    :return: The square root of CRLB in degrees/sec
    """
    w = to_rad(w)  # Convert to rads/sec since the rest of the code expects this unit
    # Creating vector w
    if w_along == 'all':
        vector_w = w / np.sqrt(3) * np.array([1, 1, 1]).reshape(3, 1)
    elif w_along == 'x':
        vector_w = w * np.array([1, 0, 0]).reshape(3, 1)
    elif w_along == 'y':
        vector_w = w * np.array([0, 1, 0]).reshape(3, 1)
    elif w_along == 'z':
        vector_w = w * np.array([0, 0, 1]).reshape(3, 1)
    else:
        raise AssertionError("Unknown w_along")

    # Calculate the CLRB and convert them back into degrees/sec
    if not is_saturated:
        return to_degree(calculate_CRLB_normal(vector_w, alpha, sig_s, sig_w, num_s, num_w)[plot_axis] ** (1 / 2))
    else:
        return to_degree(calculate_CRLB_saturated(vector_w, alpha, sig_s, sig_w, num_s, num_w, plot_axis) ** (1 / 2))
