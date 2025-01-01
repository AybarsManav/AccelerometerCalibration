import scipy.io
import matplotlib.pyplot as plt
#hard coded paths for now
eta = scipy.io.loadmat('C:/Users/konst/Desktop/TU Delft/Q2/Estimation and Detection/Project/accel_data (2)/Eta.mat').get('Eta').flatten()
theta = scipy.io.loadmat('C:/Users/konst/Desktop/TU Delft/Q2/Estimation and Detection/Project/accel_data (2)/Theta.mat').get('Theta').flatten()
y = scipy.io.loadmat('C:/Users/konst/Desktop/TU Delft/Q2/Estimation and Detection/Project/accel_data (2)/Measurements.mat').get('Noisy_Neasurements').flatten()
import numpy as np

N = 30
pos = 25
Mont = 500
g = 9.80665
sigma = 0.024
sigma = np.sqrt(sigma)

Theta = np.zeros(9)

K = np.diag(Theta[:3])

T = np.array([[1, -Theta[3], Theta[4]],
              [0, 1, -Theta[5]],
              [0, 0, 1]])
T_inv = np.linalg.inv(T)

b = Theta[6:].reshape(-1, 1)

uk_list = []
y = y.reshape(500, 3, 30 * 25)

y = np.array(y)
uk_list = np.array(uk_list)



#uk:3x1 y is now: 3x1 (will be 750*3x1)
def theta_est(y,uk):
    #print(uk)
    u0,u1,u2 = uk
    N = 9

    a1 = np.concatenate((np.array([u0,u1,u2,1]),np.zeros(N-4)))

    a2 = np.concatenate((np.zeros(4),[u1,u2,1],(np.zeros(2))))
    a3 = np.concatenate((np.zeros(N - 2),[u2, 1]))

    A = np.concatenate((a1.reshape(1,-1),a2.reshape(1,-1),a3.reshape(1,-1)),axis = 0) #A for now is 3x9 (will be 750*3x9)

    #y=np.asarray(y)
    h = np.linalg.pinv(A)@y
    h00, h01, h02, h03, h11, h12, h13, h22, h23 = h
    kx = h00; ky = h11; kz = h22; alpha_yz = h01/kx; alpha_zx = h12/ky; alpha_zy = alpha_yz*alpha_zx - h02/kx
    bx = h03;by = h13;bz=h23

    th_est = [kx,ky,kz,alpha_zx,alpha_yz,alpha_zy,bx,by,bz]
    return th_est

#mu: 3x1, eta: 2x1
def calc_u(eta):
    phi,th = eta
    return g * np.array([
        -np.sin(phi),
        np.cos(phi) * np.sin(th),
        np.cos(phi) * np.cos(th)
    ])

u = []
for i in range(pos):
    uk = calc_u(eta[i:i+2])
    u.append(uk)
print(len(u))

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