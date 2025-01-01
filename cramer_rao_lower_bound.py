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

FIM = compute_Fisher_Information_Matrix(eta, 30, sigma)
CRLB = np.linalg.inv(FIM)
# print(CRLB)

# Find lower bounds of each parameter by looking at the diagonal of the CRLB matrix
lower_bounds = np.sqrt(np.diag(CRLB))
# Also take the square root for it to be similar to the cited paper
lower_bounds = np.sqrt(lower_bounds)
print(lower_bounds)

