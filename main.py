import scipy.io

Eta = scipy.io.loadmat('C:/Users/konst/Desktop/TU Delft/Q2/Estimation and Detection/Project/accel_data (2)/Eta.mat').get('Eta')
theta = scipy.io.loadmat('C:/Users/konst/Desktop/TU Delft/Q2/Estimation and Detection/Project/accel_data (2)/Theta.mat').get('Theta')
y = scipy.io.loadmat('C:/Users/konst/Desktop/TU Delft/Q2/Estimation and Detection/Project/accel_data (2)/Measurements.mat').get('Noisy_Neasurements')
import numpy as np

N = 30
pos = 25
Mont = 500
g = 9.80665
sigma = 0.024
sigma = np.sqrt(sigma)

Theta = np.zeros(9)

K = np.diag(Theta[:3])

# Misalignment matrix (T)
T = np.array([[1, -Theta[3], Theta[4]],
              [0, 1, -Theta[5]],
              [0, 0, 1]])
T_inv = np.linalg.inv(T)

# Bias vector (b)
b = Theta[6:].reshape(-1, 1)  # 3x1 vector

print(Eta.shape)
uk_list = []
y = y.reshape(500, 3, 30 * 25)

y = np.array(y)
uk_list = np.array(uk_list)

for sim in range(Mont):
    for i in range(N*pos):
        #log likelihood is 1/sigma2 * sum (l2(yk-uk))

        #do the MLE: maximize the log likelihood over distributions
        #sum over all diff yk - muk
        #derivations?


def cost_function(Theta):
    kx, ky, kz, alpha_yz, alpha_zy, alpha_zx, bx, by, bz = theta

    K = np.diag([kx, ky, kz])
    T = np.array([[1, -alpha_yz, alpha_zy],
                  [0, 1, -alpha_zx],
                  [0, 0, 1]])
    T_inv = np.linalg.inv(T)
    b = np.array([bx, by, bz]).reshape(-1, 1)


def error(meas, mu):


