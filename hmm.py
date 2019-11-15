import numpy as np

rateOfChange = np.matrix("0.9 0.2; 0.1 0.8")
phi = np.matrix("0.1 0.3; 0.2 0.0; 0.4 0.3; 0.0 0.3; 0.3 0.1")

x_0 = np.matrix("0.6; 0.4")
T = 5



x_t = x_0
for t in range(T):
    x_t = rateOfChange * x_t

print(x_t)
py = x_t.T * phi.T

print(py)
