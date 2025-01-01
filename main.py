import scipy.io
import matplotlib.pyplot as plt
#hard coded paths for now
eta = scipy.io.loadmat('data/Eta.mat').get('Eta')
theta = scipy.io.loadmat('data/Theta.mat').get('Theta')
y = scipy.io.loadmat('data/Measurements.mat').get('Noisy_Neasurements')
import numpy as np

N = 30
pos = 25
Mont = 500
g = 9.80665
sigma = 0.024
sigma = np.sqrt(sigma)

# True parameters
K_True = np.diag(theta[:3]) # 3x3 diagonal k_x, k_y, k_z

T_true = np.array([[1, -theta[3], theta[4]],
              [0, 1, -theta[5]],
              [0, 0, 1]]) # 3x3 matrix T

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
def build_system(eta, y_single_monte_carlo):
    num_poses = eta.shape[0]
    num_observations = N
    # First compute u_k for all poses, uk_matrix will be 25x3
    uk_matrix = np.zeros((num_poses, 3))
    for i in range(num_poses):
        uk = calc_u_single_pose(eta[i])
        uk_matrix[i] = uk
    
    # Now build A and b. A will be (25*30*3)x9, b will be (25*30*3)x1
    A = np.zeros((0, 9))
    b = np.zeros((0, 1))
    for i in range(num_poses):
        for j in range(num_observations):
            A = build_A_single_iter(uk_matrix[i], A)
            b = build_b_single_iter(y_single_monte_carlo[i][j], b)
    return A, b

def solve_system(A, b):
    return np.linalg.pinv(A) @ b

def derive_theta_params(h):
    h00, h01, h02, h03, h11, h12, h13, h22, h23 = h
    kx = h00; ky = h11; kz = h22
    alpha_yz = h01 / kx; alpha_zx = h12 / ky; alpha_zy = alpha_yz * alpha_zx - h02 / kx
    bx = h03; by = h13; bz = h23
    return kx, ky, kz, alpha_zx, alpha_yz, alpha_zy, bx, by, bz

A, b = build_system(eta=eta, y_single_monte_carlo=y[0])

h = solve_system(A, b)

theta_est = derive_theta_params(h)

theta_all = []

for i in range(Mont):
    yk = y[i, :, :]
    sim_theta = []
    for mn in range(750):  # N * pos = 750
        th_est = theta_est(yk, u[mn // N])
        sim_theta.append(th_est)

    theta_all.append(sim_theta)

theta_all = np.array(theta_all)

print(len(theta_all))

mse_values = np.zeros((Mont, N * pos))

for i in range(Mont):
    for j in range(N * pos):
        mse_values[i, j] = np.mean((theta_all[i, j, :] - theta) ** 2)

mean_mse = np.mean(mse_values, axis=0)
std_mse = np.std(mse_values, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, N * pos + 1), mean_mse, label='Mean MSE')
plt.fill_between(np.arange(1, N * pos + 1), mean_mse - std_mse, mean_mse + std_mse, color='b', alpha=0.2, label='±1 Standard Deviation')
plt.title('Mean MSE of Estimated Parameters (Theta) vs True Parameters')
plt.xlabel('Number of Samples')
plt.ylabel('Mean MSE')
plt.legend()
plt.grid(True)
plt.show()