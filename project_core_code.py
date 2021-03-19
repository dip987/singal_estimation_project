import math
import pandas as pd
import numpy as np
from pathlib import Path
from helper_functions import skew, calculate_CRLB_normal, calculate_CRLB_saturated, A, rmse, generate_phi, to_rad, \
    to_degree
from numpy.linalg import inv
from data_classes import GeneratedData, Params
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


# Working with frame config 4(a)
# Our r matrix, gives the position of each sensor w.r.t. the center


def generate(w: np.ndarray, phi: np.ndarray, noise_s_vector: np.ndarray, noise_w_vector: np.ndarray):
    # Gyroscope Model
    y_w = np.concatenate([w] * Params.num_w, axis=0) + noise_w_vector
    average_w = np.average(y_w.reshape(3, -1, order='F'), axis=1)

    # Check for saturation and remove the corresponding data rows
    saturation_point = to_rad(2000)

    # Figure out which rows are saturated and remove them from the data
    is_saturated: bool = False
    num_saturated_w = 0
    saturated_indices = []
    for index in range(len(y_w)):
        # Saturation
        if abs(y_w[index]) >= saturation_point:
            saturated_indices.append(index)
            num_saturated_w += 1
            is_saturated = True
        else:
            pass
    y_w = np.delete(y_w, saturated_indices, axis=0)

    # Accelerometer Model
    temp_a = -1 * np.concatenate([skew(r_i) for r_i in Params.r], axis=0)
    temp_b = np.concatenate([np.identity(3)] * Params.num_s, axis=0)
    Hs = np.concatenate([temp_a, temp_b], axis=1)

    omega_w_2 = skew(w) @ skew(w)
    h_s = np.concatenate([omega_w_2 @ r_i.reshape(3, 1) for r_i in Params.r], axis=0)
    y_s = h_s + Hs @ phi + noise_s_vector

    # Combined Model
    output = GeneratedData(y_s=y_s, y_w=y_w, h_s=h_s, Hs=Hs, r=Params.r, is_saturated=is_saturated,
                           num_saturated_w=num_saturated_w, saturated_indices=saturated_indices, average_w=average_w)
    return output


# Estimation
def estimate(data: GeneratedData, sig_w, sig_s, calculate_tensor_estimate: bool = False):
    # Generate the Covariance Matrix
    Qs = np.diag([sig_s ** 2] * len(data.y_s))  # In meters
    Qw = np.diag([sig_w ** 2] * len(data.y_w))  # In rad/s
    Q = np.diag([sig_s ** 2] * len(data.y_s) + [sig_w ** 2] * len(data.y_w))
    tensor_w = None

    # Figure out w first using iterations of Newton-Gauss Method then use that w to figure out phi
    if not data.is_saturated:
        iteration_steps = 5  # Unsaturated Data converges faster

        temp_a = np.concatenate([np.identity(3)] * Params.num_w, axis=1)
        temp_b = np.concatenate([np.identity(3)] * Params.num_w, axis=0)
        current_w_estimate = inv(temp_a @ inv(Qw) @ temp_b) @ temp_a @ inv(Qw) @ data.y_w

        P = inv(Q) - inv(Q) @ data.H @ inv(data.H.T @ inv(Q) @ data.H) @ data.H.T @ inv(Q)

    else:
        iteration_steps = 20  # Saturated Data takes longer to converges
        temp_a = np.concatenate([np.identity(3)] * Params.num_w, axis=1)
        temp_a = np.delete(temp_a, data.saturated_indices, axis=1)
        temp_b = np.concatenate([np.identity(3)] * Params.num_w, axis=0)
        temp_b = np.delete(temp_b, data.saturated_indices, axis=0)
        current_w_estimate = np.array([2000, -1, -1], dtype=float).reshape(3, 1)
        P = inv(Q) - inv(Q) @ data.H @ inv(data.H.T @ inv(Q) @ data.H) @ data.H.T @ inv(Q)

    # Iterations for estimating w
    for iteration in range(iteration_steps):
        Jh = np.concatenate([A(current_w_estimate, r_i.T).T for r_i in Params.r] + [temp_a], axis=1).T
        current_w_estimate += inv(Jh.T @ P @ Jh) @ Jh.T @ P @ (data.y - data.h(current_w_estimate))
    # Use the estimated w to determine phi
    phi_estimate = inv(data.H.T @ inv(Q) @ data.H) @ data.H.T @ inv(Q) @ (data.y - data.h(current_w_estimate))

    # Tensor Estimate
    # PS: Tensor estimate only works when we have IMUs along all 3 axes. Otherwise the calculations would result in
    # trying to invert a singular matrix. Make sure that is not the case before allowing [calculate_tensor_estimate]
    if calculate_tensor_estimate:
        # For simplicity we are assuming the tensor estimate as positive. Usually the sign of the tensor estimate is
        # determined from the gyroscope data/previous values. But for our simulations, the we are only taking a single
        # angular velocity at a time. There are no previous values and the gyroscopes doesn't affect CRLB. So those are
        # ignored. It is by default set to give positive angular velocity
        R = np.concatenate([np.array([1] * len(data.r)).reshape(1, -1), data.r.T], axis=0)
        X_hat = data.y_s.reshape(3, -1, order='F') @ R.T @ inv(R @ R.T)  # Column major reshaping
        W_hat = X_hat[:, 1:]
        w_wT = (W_hat + W_hat.T) / 2 - 1 / 4 * np.trace(W_hat + W_hat.T) * np.identity(3)
        tensor_w_2 = np.array([w_wT[0, 0], w_wT[1, 1], w_wT[2, 2]]).reshape(3, 1)
        tensor_w = np.sqrt(np.abs(tensor_w_2))

    return current_w_estimate, phi_estimate, tensor_w


# Won't change
phi_guess = generate_phi(np.array([0, 0, 9.8]).reshape(3, 1), 0 * np.ones((3, 1)))

######### TEST CODE ################
# w = np.array([500 * np.pi / 180, 0, 0]).reshape(3, 1)
# tries = 1000
# # Store estimate for each random trial
# all_w_estimates = np.zeros((3, tries))
# all_average_w = np.zeros((3, tries))
# all_tensor_w_estimate = np.zeros((3, tries))
# sig_s = 0.01
# sig_w = 1 * np.pi / 180
# for i in range(tries):
#     # Generate a set of random data and estimate omega from them
#     # Zero mean, std_dev = scale
#     n_s = np.random.normal(loc=0, scale=sig_s, size=(Params.num_s * 3, 1))
#     n_w = np.random.normal(loc=0, scale=sig_w, size=(Params.num_w * 3, 1))
#     generated_data = generate(w, phi_guess, n_s, n_w)
#     w_estimate, phi, tensor_w_estimate = estimate(generated_data, sig_w, sig_s, calculate_tensor_estimate=True)
#     all_w_estimates[:, i] = w_estimate.flatten()
#     all_average_w[:, i] = generated_data.average_w.flatten()
#     all_tensor_w_estimate[:, i] = tensor_w_estimate.flatten()
#
# print(f'proposed estimate : {rmse(true_w=w, estimated_w=all_w_estimates) * 180 / np.pi}')
# print(f'averaging : {rmse(true_w=w, estimated_w=all_average_w) * 180 / np.pi}')
# print(f'tensor acceleration estimate : {rmse(true_w=w, estimated_w=all_tensor_w_estimate) * 180 / np.pi}')

######### TEST CODE ################


#### Code for Curve ####

# rmse_proposed_method = []
# rmse_averaging = []
# rmse_tensor_method = []
# w_space = np.logspace(2, 4, 20)  # w in degrees/sec
# # Frequencies to plot
# for abs_w in w_space:
#     abs_w_rad = abs_w * np.pi / 180
#     # Change with every iteration
#     w = abs_w_rad / np.sqrt(3) * np.array([1, 1, 1]).reshape(3, 1)
#
#     monte_carlo_trials = 1000
#     # Store estimate for each random trial
#     all_w_estimates = np.zeros((3, monte_carlo_trials))
#     all_average_w = np.zeros((3, monte_carlo_trials))
#     all_tensor_w_estimate = np.zeros((3, monte_carlo_trials))
#     sig_s = 0.01
#     sig_w = 1 * np.pi / 180
#     for i in range(monte_carlo_trials):
#         # Generate a set of random data and estimate omega from them
#         # Zero mean, std_dev = scale
#         n_s = np.random.normal(loc=0, scale=sig_s, size=(Params.num_s * 3, 1))
#         n_w = np.random.normal(loc=0, scale=sig_w, size=(Params.num_w * 3, 1))
#         generated_data = generate(w, phi_guess, n_s, n_w)
#         w_estimate, phi, tensor_w_estimate = estimate(generated_data, sig_w, sig_s, calculate_tensor_estimate=True)
#         all_w_estimates[:, i] = w_estimate.flatten()
#         all_average_w[:, i] = generated_data.average_w.flatten()
#         all_tensor_w_estimate[:, i] = tensor_w_estimate.flatten()
#
#     rmse_proposed_method.append(rmse(true_w=w, estimated_w=all_w_estimates) * 180 / np.pi)
#     rmse_tensor_method.append(rmse(true_w=w, estimated_w=all_tensor_w_estimate) * 180 / np.pi)
#     rmse_averaging.append(rmse(true_w=w, estimated_w=all_average_w) * 180 / np.pi)
#
# rmse_averaging = np.array(rmse_averaging)
# rmse_proposed_method = np.array(rmse_proposed_method)
# rmse_tensor_method = np.array(rmse_tensor_method)
#
# plt.figure()
# smooth_w_space = np.linspace(w_space[0], w_space[-1], 30)
# for index in range(3):
#     data = rmse_averaging[:, index]
#     model = make_interp_spline(w_space, data, k=2)
#     smooth_data = model(smooth_w_space)
#     plt.plot(smooth_w_space, smooth_data)
#
# for index in range(3):
#     data = rmse_proposed_method[:, index]
#     model = make_interp_spline(w_space, data, k=2)
#     smooth_data = model(smooth_w_space)
#     plt.plot(smooth_w_space, smooth_data)
#
# for index in range(3):
#     data = rmse_tensor_method[:, index]
#     model = make_interp_spline(w_space, data, k=2)
#     smooth_data = model(smooth_w_space)
#     plt.plot(smooth_w_space, smooth_data)
#
# plt.legend(['w_x average', 'w_y average', 'w_z average', 'w_x MLE', 'w_y MLE', 'w_z MLE', 'tensor w_x', 'tensor w_y',
#             'tensor w_z'])
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel('Angular Velocity in degrees/sec')
# plt.ylabel('RMSE')
# plt.grid(which='both')
# plt.show()
