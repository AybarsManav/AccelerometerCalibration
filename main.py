import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from cramer_rao_lower_bound import compute_CRLB_for_num_observations, plot_CRLB_for_different_variences

eta = scipy.io.loadmat('data/Eta.mat').get('Eta')
theta = scipy.io.loadmat('data/Theta.mat').get('Theta')
y = scipy.io.loadmat('data/Measurements.mat').get('Noisy_Neasurements')

N = 30
pos = 25
Mont = 500
g = 9.80665
sigma = np.sqrt(0.024)

# True parameters
K_True = np.diag(theta[:3]) # 3x3 diagonal k_x, k_y, k_z


b_true = theta[6:].reshape(-1, 1) # 3x1 bias vector

# 25 x 2 rotational angles for input
eta = eta.reshape(-1, 2) # 25x2

y = y.transpose(0, 3, 2, 1) # 500x25x30x3
y = np.array(y)

def build_A_single_iter(uk, A):
    u0, u1, u2 = uk
    a1 = np.concatenate((np.array([u0, u1, u2, 1]), np.zeros(5))).reshape(1, -1)
    a2 = np.concatenate((np.zeros(4), [u1, u2, 1], np.zeros(2))).reshape(1, -1)
    a3 = np.concatenate((np.zeros(7), [u2, 1])).reshape(1, -1)
    A = np.concatenate((A, a1, a2, a3), axis=0)
    return A

def build_b_single_iter(yk, b):
    b = np.concatenate((b, yk.reshape(3, 1)), axis=0)
    return b

#Takes in a single row of eta
# u: 3x1, eta: 2x1
def calc_u_single_pose(eta):
    phi,th = eta
    return g * np.array([
        -np.sin(phi),
        np.cos(phi) * np.sin(th),
        np.cos(phi) * np.cos(th)
    ])

# Builds Ax=b system for a single monte carlo iteration
def build_system(eta, y_single_monte_carlo, num_observations=30):
    num_poses = eta.shape[0]
    # First compute u_k for all poses, uk_matrix will be 25x3
    uk_matrix = np.zeros((num_poses, 3))
    for i in range(num_poses):
        uk = calc_u_single_pose(eta[i])
        uk_matrix[i] = uk
    
    # Now build A and b. A will be (25*num_observations*3)x9, b will be (25*num_observations*3)x1
    A = np.zeros((0, 9))
    b = np.zeros((0, 1))
    for i in range(num_poses):
        for j in range(num_observations):
            A = build_A_single_iter(uk_matrix[i], A)
            b = build_b_single_iter(y_single_monte_carlo[i][j], b)
    return A, b

def solve_system(A, b):
    return np.linalg.pinv(A) @ b
def solve_BLUE(H, y):
    return np.linalg.pinv(H.T@H)@H.T@y

def derive_theta_params(h):
    h00, h01, h02, h03, h11, h12, h13, h22, h23 = h
    kx = h00; ky = h11; kz = h22
    alpha_yz = h01 / kx; alpha_zx = h12 / ky; alpha_zy = alpha_yz * alpha_zx - h02 / kx
    bx = h03; by = h13; bz = h23
    return kx, ky, kz, alpha_yz, alpha_zy, alpha_zx, bx, by, bz

def estimate_theta_for_single_monte_carlo(eta, y_single_monte_carlo, num_observations=30, type = 'MLE'):
    A, b = build_system(eta, y_single_monte_carlo, num_observations)
    if type == 'MLE':
        h = solve_system(A, b)
    elif type == 'BLUE':
        h = solve_BLUE(A, b)
    return derive_theta_params(h)

# Find estimated parameters for different monte carlo iterations and using different number of observations
num_of_observations_vec = np.arange(1, 31)
num_monte_carlo_iters = 500 # Since Mont = 500 takes too long debugging with this first.
all_theta_est = np.zeros((num_monte_carlo_iters, num_of_observations_vec.shape[0], 9))
for i in range(num_monte_carlo_iters):
    for j in num_of_observations_vec:
        all_theta_est[i, j - 1] = np.array(estimate_theta_for_single_monte_carlo(eta, y[i, :, :], j,type = 'MLE')).flatten()

def computeRMSE(theta_est, theta):
    return np.sqrt(np.mean((theta_est - theta) ** 2, axis=0))

# Compute RMSE for all monte carlo iterations and number of observations
all_theta_est = all_theta_est.transpose(1, 0, 2) # 30x500x9
rmse_values = np.zeros((num_of_observations_vec.shape[0], 9))
for i in range(num_of_observations_vec.shape[0]):
        rmse_values[i] = computeRMSE(all_theta_est[i, :], theta.transpose())

# Compute the average value and standard deviation of the estimated parameters
# Use all N=30 observations
avg_theta_est = np.mean(all_theta_est[-1, :, :], axis=0).reshape(9, 1)
std_theta_est = np.std(all_theta_est[-1, :, :], axis=0).reshape(9, 1)

# Make a table showing the true values, average estimated values, and standard deviation of the estimated values
table = np.concatenate((theta, avg_theta_est, std_theta_est), axis=1)
print("True Values, Average Estimated Values, Standard Deviation of Estimated Values")
print(table)

# Estimate CRLB for different number of observations
CRLB_vec = compute_CRLB_for_num_observations(eta, num_of_observations_vec, sigma)

# Plot RMSE values for k parameters
plt.figure(figsize=(10, 10))
plt.plot(num_of_observations_vec, rmse_values[:, 0], label=r'$\sqrt{MSE_{k_x}}$', color='b')
plt.plot(num_of_observations_vec, rmse_values[:, 1], label=r'$\sqrt{MSE_{k_y}}$', color='g')
plt.plot(num_of_observations_vec, rmse_values[:, 2], label=r'$\sqrt{MSE_{k_z}}$', color='r')
# Also plot CRLB values for k parameters
plt.plot(num_of_observations_vec, CRLB_vec[:, 0], label=r'$\sqrt{CRLB_{k_x}}$', linestyle='--', color='b')
plt.plot(num_of_observations_vec, CRLB_vec[:, 1], label=r'$\sqrt{CRLB_{k_y}}$', linestyle='--', color='g')
plt.plot(num_of_observations_vec, CRLB_vec[:, 2], label=r'$\sqrt{CRLB_{k_z}}$', linestyle='--', color='r')
plt.xlabel("Number of Observations")
plt.ylabel(r'$\sqrt{MSE}$')
plt.title(r'$\sqrt{MSE}$ vs Number of Observations for k parameters')
plt.legend()
plt.grid(True)

# Plot sqrt MSE values for alpha parameters
plt.figure(figsize=(10, 10))
plt.plot(num_of_observations_vec, rmse_values[:, 3], label=r'$\sqrt{MSE_{\alpha_{yz}}}$', color='b')
plt.plot(num_of_observations_vec, rmse_values[:, 4], label=r'$\sqrt{MSE_{\alpha_{zy}}}$', color='g')
plt.plot(num_of_observations_vec, rmse_values[:, 5], label=r'$\sqrt{MSE_{\alpha_{zx}}}$', color='r')
# Also plot CRLB values for alpha parameters
plt.plot(num_of_observations_vec, CRLB_vec[:, 3], label=r'$\sqrt{CRLB_{\alpha_{yz}}}$', linestyle='--', color='b')
plt.plot(num_of_observations_vec, CRLB_vec[:, 4], label=r'$\sqrt{CRLB_{\alpha_{zy}}}$', linestyle='--', color='g')
plt.plot(num_of_observations_vec, CRLB_vec[:, 5], label=r'$\sqrt{CRLB_{\alpha_{zx}}}$', linestyle='--', color='r')
plt.xlabel("Number of Observations")
plt.ylabel(r'$\sqrt{MSE}$')
plt.title(r'$\sqrt{MSE}$ vs Number of Observations for $\alpha$ parameters')
plt.legend()
plt.grid(True)

# Plot sqrt MSE values for b parameters
plt.figure(figsize=(10, 10))
plt.plot(num_of_observations_vec, rmse_values[:, 6], label=r'$\sqrt{MSE_{b_x}}$', color='b')
plt.plot(num_of_observations_vec, rmse_values[:, 7], label=r'$\sqrt{MSE_{b_y}}$', color='g')
plt.plot(num_of_observations_vec, rmse_values[:, 8], label=r'$\sqrt{MSE_{b_z}}$', color='r')
# Also plot CRLB values for b parameters
plt.plot(num_of_observations_vec, CRLB_vec[:, 6], label=r'$\sqrt{CRLB_{b_x}}$', linestyle='--', color='b')
plt.plot(num_of_observations_vec, CRLB_vec[:, 7], label=r'$\sqrt{CRLB_{b_y}}$', linestyle='--', color='g')
plt.plot(num_of_observations_vec, CRLB_vec[:, 8], label=r'$\sqrt{CRLB_{b_z}}$', linestyle='--', color='r')
plt.xlabel("Number of Observations")
plt.ylabel(r'$\sqrt{MSE}$')
plt.title(r'$\sqrt{MSE}$ vs Number of Observations for b parameters')

plt.legend()
plt.grid(True)

# Plot CRLB for different standard deviations
standard_dev_vec = np.linspace(0.01, 0.1, 100)
plot_CRLB_for_different_variences(eta, 30, standard_dev_vec)