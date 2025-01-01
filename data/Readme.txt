N=30                         % Number of observations in each position
Num_Position = 25;           % Number of positions 
Mont = 500                   % Number of Monte Carlo 
g = 9.80665 [m/s^2]
sigma^2 = 0.024 [m/s^2]      % Variance of noise 


Eta = 50x1 vector of rotational angles in radian. 
Theta = 9x1 vector of true parameters 


Noisy_Neasurements                 % 500x3x30x25  Measurements data 
Reshape the Noisy_Neasurements matrix into a vector form as in (7) of assignment for each Monte Carlo Simulation.  


Python programmers can use 

import scipy.io
mat = scipy.io.loadmat('filename.mat')

to load .mat file in python.  