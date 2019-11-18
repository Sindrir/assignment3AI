import numpy as np

theta = np.matrix("0.9 0.2; 0.1 0.8")
phi = np.matrix("0.1 0.3; 0.2 0.0; 0.4 0.3; 0.0 0.3; 0.3 0.1")

x_0 = np.matrix("0.6 0.4")
T = 5
y = [3, 3, 0, 4, 2]
#y = [0, 0, 0, 0, 0]
#y = [4]

def hmm(theta, phi, x_0, y, T):
    # Hidden Markov Model function
    # Theta is the transition probability table
    # Phi is the observation probability table
    # x_0 is the starting probability of happiness and sadness
    # y is a list of activities
    # T is the count of activites/timesteps
    xT = np.zeros(shape=(6,2))
    xT[0] = x_0
    for t in range(T+1):
        if t != 0:
            sTMinus = ((phi[y[t-1], 1] * xT[t-1,1]) / ((phi[y[t-1], 1] * xT[t-1,1]) + (phi[y[t-1], 0] * xT[t-1,0])))
            hTMinus = ((phi[y[t-1], 0] * xT[t-1,0]) / ((phi[y[t-1], 0] * xT[t-1,0]) + (phi[y[t-1], 1] * xT[t-1,1])))
            top = (phi[y[t-1], 0] * (0.2 * sTMinus + 0.9 * hTMinus))
            bottom = (phi[y[t-1], 0] * (0.2 * sTMinus + 0.9 * hTMinus)) + (phi[y[t-1], 1] * (1 - (0.2 * sTMinus + 0.9 * hTMinus)))
            xT[t,0] = top / bottom # probability of happiness
            xT[t,1] = 1 - xT[t,0]  # probability of sadness
    return xT

print(hmm(theta, phi, x_0, y, T))
