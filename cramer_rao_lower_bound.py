import scipy.io
import matplotlib.pyplot as plt
import numpy as np

eta = scipy.io.loadmat('data/Eta.mat').get('Eta')
theta = scipy.io.loadmat('data/Theta.mat').get('Theta')
y = scipy.io.loadmat('data/Measurements.mat').get('Noisy_Neasurements')

N = 30
pos = 25
Mont = 500
g = 9.80665
sigma2 = 0.01
sigma = np.sqrt(sigma2)

# 25 x 2 rotational angles for input
eta = eta.reshape(-1, 2) # 25x2

#Takes in a single row of eta
# u: 3x1, eta: 2x1
def calc_u_single_pose(eta):
    phi,th = eta
    return g * np.array([
        -np.sin(phi),
        np.cos(phi) * np.sin(th),
        np.cos(phi) * np.cos(th)
    ], dtype=np.float64)

k_x, k_y, k_z = theta[:3]
alpha_yz, alpha_zy, alpha_zx = theta[3:6]
b_x, b_y, b_z = theta[6:]

def delta_k_x(uk):
    return np.array([
        uk[0] + uk[1] * alpha_yz + uk[2] * (alpha_yz * alpha_zx - alpha_zy),
        0,
        0
    ], dtype=np.float64)

def delta_k_y(uk):
    return np.array([
        0,
        uk[1] + uk[2] * alpha_zx,
        0
    ], dtype=np.float64)

def delta_k_z(uk):
    return np.array([
        0,
        0,
        uk[2]
    ], dtype=np.float64)

def delta_alpha_yz(uk):
    return np.array([
        uk[1] * k_x + uk[2] * k_x * alpha_zx,
        0,
        0
    ], dtype=np.float64)

def delta_alpha_zy(uk):
    return np.array([
        -uk[2] * k_x,
        0,
        0
    ], dtype=np.float64)

def delta_alpha_zx(uk):
    return np.array([
        uk[2] * k_x * alpha_yz,
        uk[2] * k_y,
        0
    ], dtype=np.float64)

delta_b_x = np.array([1, 0, 0], dtype=np.float64)
delta_b_y = np.array([0, 1, 0], dtype=np.float64)
delta_b_z = np.array([0, 0, 1], dtype=np.float64)

def compute_Fisher_Information_Matrix(eta, num_observations_per_pose, standard_dev):
    num_poses = eta.shape[0]
    FIM = np.zeros((9, 9))
    for i in range(num_poses):
        uk = calc_u_single_pose(eta[i])
        del_mu_del_theta_i = np.array([
            delta_k_x(uk),
            delta_k_y(uk),
            delta_k_z(uk),
            delta_alpha_yz(uk),
            delta_alpha_zy(uk),
            delta_alpha_zx(uk),
            delta_b_x,
            delta_b_y,
            delta_b_z
        ])
        del_mu_del_theta_j = del_mu_del_theta_i.transpose()
        FIM += del_mu_del_theta_i @ del_mu_del_theta_j
    FIM = num_observations_per_pose * FIM / (standard_dev ** 2)
    return FIM

# FIM = compute_Fisher_Information_Matrix(eta, 30, sigma)
# print(FIM)
# CRLB = np.linalg.inv(FIM)
# # Find lower bounds of each parameter by looking at the diagonal of the CRLB matrix
# lower_bounds = np.sqrt(np.diag(CRLB))

def compute_CRLB_for_num_observations(eta, num_observations_vec, standard_dev):
    CRLB_vec = np.zeros((len(num_observations_vec), 9))
    for num_observations in num_observations_vec:
        FIM = compute_Fisher_Information_Matrix(eta, num_observations, standard_dev)
        CRLB = np.linalg.inv(FIM)
        np.sqrt(np.diag(CRLB))
        CRLB_vec[num_observations - 1] = np.sqrt(np.diag(CRLB))
    return CRLB_vec

def plot_CRLB_for_different_variences(eta, num_observations, standard_dev_vec):
    CRLB_vec = np.zeros((len(standard_dev_vec), 9))
    for i, standard_dev in enumerate(standard_dev_vec):
        FIM = compute_Fisher_Information_Matrix(eta, num_observations, standard_dev)
        CRLB = np.linalg.inv(FIM)
        CRLB_vec[i] = np.diag(CRLB) # Plot the variance instead of standard deviation #np.sqrt(np.diag(CRLB))

    # Plot the CRLB for grouped parameters in the range of standard deviations
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    fig3, ax3 = plt.subplots(figsize=(12, 7))

    # Plot k_x, k_y, k_z in the first figure
    ax1.plot(standard_dev_vec ** 2, CRLB_vec[:, 0], label="$CRLB_{k_x}$", color='b')
    ax1.plot(standard_dev_vec ** 2, CRLB_vec[:, 1], label="$CRLB_{k_y}$", color='g')
    ax1.plot(standard_dev_vec ** 2, CRLB_vec[:, 2], label="$CRLB_{k_z}$", color='r')
    ax1.set_title('$CRLB$ for $k_x$, $k_y$, $k_z$', fontsize=22)
    ax1.set_xlabel('Noise Variance ($\\sigma^2$)', fontsize=20)
    ax1.set_ylabel('$CRLB$', fontsize=20)
    ax1.legend(fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.grid(True)

    # Plot alpha_yz, alpha_zy, alpha_zx in the second figure
    ax2.plot(standard_dev_vec ** 2, CRLB_vec[:, 3], label="$CRLB_{\\alpha_{yz}}$", color='b')
    ax2.plot(standard_dev_vec ** 2, CRLB_vec[:, 4], label="$CRLB_{\\alpha_{zy}}$", color='g')
    ax2.plot(standard_dev_vec ** 2, CRLB_vec[:, 5], label="$CRLB_{\\alpha_{zx}}$", color='r')
    ax2.set_title('$CRLB$ for $\\alpha_{yz}$, $\\alpha_{zy}$, $\\alpha_{zx}$', fontsize=22)
    ax2.set_xlabel('Noise Variance ($\\sigma^2$)', fontsize=20)
    ax2.set_ylabel('$CRLB$', fontsize=20)
    ax2.legend(fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.grid(True)

    # Plot b_x, b_y, b_z in the third figure
    ax3.plot(standard_dev_vec ** 2, CRLB_vec[:, 6], label="$CRLB_{b_x}$", color='b')
    ax3.plot(standard_dev_vec ** 2, CRLB_vec[:, 7], label="$CRLB_{b_y}$", color='g')
    ax3.plot(standard_dev_vec ** 2, CRLB_vec[:, 8], label="$CRLB_{b_z}$", color='r')
    ax3.set_title('CRLB for $b_x$, $b_y$, $b_z$', fontsize=22)
    ax3.set_xlabel('Noise Variance ($\\sigma^2$)', fontsize=20)
    ax3.set_ylabel('CRLB', fontsize=20)
    ax3.legend(fontsize=18)
    ax3.tick_params(axis='both', which='major', labelsize=18)
    ax3.grid(True)

    plt.show()


