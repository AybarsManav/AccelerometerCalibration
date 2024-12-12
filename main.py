import scipy.io

eta = scipy.io.loadmat('C:/Users/konst/Desktop/TU Delft/Q2/Estimation and Detection/Project/accel_data (2)/Eta.mat').get('Eta')
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

T = np.array([[1, -Theta[3], Theta[4]],
              [0, 1, -Theta[5]],
              [0, 0, 1]])
T_inv = np.linalg.inv(T)

b = Theta[6:].reshape(-1, 1)

uk_list = []
y = y.reshape(500, 3, 30 * 25)

y = np.array(y)
uk_list = np.array(uk_list)

#for sim in range(Mont):
#   for i in range(N*pos):
        #log likelihood is 1/sigma2 * sum (l2(yk-uk))

        #do the MLE: maximize the log likelihood over distributions
        #sum over all diff yk - muk
        #derivations?

#uk:3x1 y is now: 3x1 (will be 750*3x1)
def theta_est(y,uk):
    [u0,u1,u2] = uk
    print(u0,u1,u2)
    N = len(theta)

    a1 = np.concatenate(np.array([u0,u1,u2,1]),(np.zeros(N-4)))
    a2 = np.concatenate((np.zeros(4),[u1,u2,1],(np.zeros(2))))
    a3 = np.concatenate((np.zeros(N - 2),[u2, 1]))
    A = np.concatenate((a1,a2,a3),axis = 0) #A for now is 3x9 (will be 750*3x9)

    h = np.linalg.pinv(A)@y
    h00, h01, h02, h03, h10, h11, h12, h13, h20, h21, h22, h23 = h
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

u = calc_u(eta[0:2])
th_est = theta_est(y[0,:,0],u)
print(th_est)
