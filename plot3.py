from data_classes import Params
from project_core_code import generate, estimate
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import crlb_plot_wrapper, generate_phi, rmse, to_rad, to_degree

# Working with frame config 4(b)
# Our r matrix, gives the position of each sensor w.r.t. the center
r = np.array([[0.01, 0, 0],
              [0, 0.01, 0],
              [-0.01, 0, 0],
              [0, -0.01, 0],
              [0, 0, 0.01],
              [0, 0, -0.01]])
Params.num_w = 6
Params.num_s = 6
Params.r = r
sig_s = 0.01
sig_w = to_rad(1.)

w_space = np.logspace(2, 4, 20)
# w_space = np.array([2000, 3001])
# Won't change
phi_guess = generate_phi(np.array([0, 0, 0]).reshape(3, 1), 0 * np.ones((3, 1)))

#### Code for Monte Carlo####
monte_carlo_trials = 500
rmse_proposed_method = []
rmse_averaging = []
rmse_tensor_method = []
# Taking w between 10^2 to 10^4 with 10 points
# Frequencies to plot
for abs_w in w_space:
    abs_w_rad = to_rad(abs_w)
    # Change with every iteration
    w = abs_w_rad / np.sqrt(3) * np.array([1, 1, 1]).reshape(3, 1)

    # Store estimate for each random trial
    all_w_estimates = np.zeros((3, monte_carlo_trials))
    all_average_w = np.zeros((3, monte_carlo_trials))
    all_tensor_w_estimate = np.zeros((3, monte_carlo_trials))
    for i in range(monte_carlo_trials):
        # Generate a set of random data and estimate omega from them
        # Zero mean, std_dev = scale
        n_s = np.random.normal(loc=0, scale=sig_s, size=(Params.num_s * 3, 1))
        n_w = np.random.normal(loc=0, scale=sig_w, size=(Params.num_w * 3, 1))
        generated_data = generate(w, phi_guess, n_s, n_w)
        # Estimate
        w_estimate, phi, tensor_w_estimate = estimate(generated_data, sig_w, sig_s, calculate_tensor_estimate=True)
        all_w_estimates[:, i] = w_estimate.flatten()
        all_average_w[:, i] = generated_data.average_w.flatten()
        all_tensor_w_estimate[:, i] = tensor_w_estimate.flatten()

    rmse_proposed_method.append(to_degree(rmse(true_w=w, estimated_w=all_w_estimates)))
    rmse_tensor_method.append(to_degree(rmse(true_w=w, estimated_w=all_tensor_w_estimate)))
    rmse_averaging.append(to_degree(rmse(true_w=w, estimated_w=all_average_w)))

rmse_averaging = np.array(rmse_averaging)
rmse_proposed_method = np.array(rmse_proposed_method)
rmse_tensor_method = np.array(rmse_tensor_method)

# Monte Carlo

plt.figure()
smooth_w_space = np.linspace(w_space[0], w_space[-1], 30)

# Plot CRLB
plt.plot(smooth_w_space,
         [crlb_plot_wrapper(w, 'all', 0.01, sig_s, sig_w, Params.num_s, Params.num_w, 0, False) for w in
          smooth_w_space], linewidth=4)
plt.plot(smooth_w_space,
         [crlb_plot_wrapper(w, 'all', 0.01, sig_s, sig_w, Params.num_s, Params.num_w, 2, True) for w in
          smooth_w_space], linewidth=4, linestyle='dashdot')
plt.plot(w_space, rmse_proposed_method[:, 0], 'o')
plt.plot(w_space, rmse_proposed_method[:, 1], 'x')
plt.plot(w_space, rmse_proposed_method[:, 2], '^')

plt.plot(w_space, rmse_tensor_method[:, 0], '<')
plt.plot(w_space, rmse_tensor_method[:, 1], '>')
plt.plot(w_space, rmse_tensor_method[:, 2], '1')


plt.legend(['sqrt(CRLB)', 'sqrt(saturated CRLB)', '$w_x MLE$', '$w_y MLE$', '$w_z MLE$', 'tensor w_x', 'tensor w_y',
            'tensor w_z'])
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Angular Velocity in degrees/sec')
plt.ylabel('RMSE')
plt.ylim(0.1, 1)
plt.grid(which='both')
plt.axvline(x=2000*np.sqrt(3), linestyle='--')
plt.show()
