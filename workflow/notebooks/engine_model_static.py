import numpy as np
import scipy as sp
from scipy.optimize import minimize

import mpmath
from mpmath import mp, mpf
mp.dps = 80
mp.pretty = True

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm
import matplotlib.colors as colors

from tqdm import tqdm_notebook as tqdm
from multiprocessing import Pool

import pickle


def Q_fn(param, k_bT=1):
    """
    Returns the 12x12 transition rate matrix of the allosteric enzyme
    for the given set of model parameter
    """
    
    # Unpack the parameters
    [dG_1, dG_2, amplitude, alpha_21, alpha_54, alpha_32, alpha_65, loffA, loffI, kA, kI, kAL, kAS, lonI, lonA, gamma] = param

    oscMidpoint = -1/2 * (-dG_1 + dG_2 - k_bT * np.log((alpha_21 * alpha_32 + alpha_21 * gamma) / (alpha_32 * gamma)))
    epsilon_1 = oscMidpoint + amplitude
    epsilon_2 = oscMidpoint - amplitude
    alpha_23 = alpha_32 * mpmath.exp((dG_2 - epsilon_1) / k_bT)
    alpha_12 = alpha_21 * mpmath.exp((dG_1 + epsilon_1) / k_bT)

    alpha_56 = alpha_65 * mpmath.exp((dG_2 - epsilon_2) / k_bT)
    alpha_45 = alpha_54 * mpmath.exp((dG_1 + epsilon_2) / k_bT)

    # Calculate the kIL value from the cycle condition
    kIL = kAL * (kI/kA) * (lonI/lonA) * (loffA/loffI)    
    
    # Calculate the values of kIS, kASL & kISL from the cycle condition
    delta = (kAS/kA)
    kIS = delta * (alpha_21/alpha_12) * (alpha_45/alpha_54) * kI
    kASL = delta * kAL
    kISL = delta * (alpha_21/alpha_12) * (alpha_45/alpha_54) * kASL
    

    #    Q_no_ligand   | matUnbinding
    #    -------------- -------------
    #    matBinding_L  |   Q_ligand
    
    Q_no_ligand = np.array([[0, kA, alpha_21, 0, gamma, 0], \
                            [kI, 0, 0, alpha_54, 0, gamma],\
                            [alpha_12, 0, 0, kAS, alpha_32, 0], \
                            [0, alpha_45, kIS, 0, 0, alpha_65], \
                            [0, 0, alpha_23, 0, 0, kAS], \
                            [0, 0, 0, alpha_56, kIS, 0]])
    
    Q_ligand = np.array([[0, kAL, alpha_21, 0, gamma, 0], \
                         [kIL, 0, 0, alpha_54, 0, gamma],\
                         [alpha_12, 0, 0, kASL, alpha_32, 0], \
                         [0, alpha_45, kISL, 0, 0, alpha_65], \
                         [0, 0, alpha_23, 0, 0, kASL], \
                         [0, 0, 0, alpha_56, kISL, 0]])
    
    matUnbinding = np.diag([loffA, loffI, loffA, loffI, loffA, loffI])
    
    Q_fixed_row1 = np.concatenate([Q_no_ligand, matUnbinding], axis = 1)
    
    matBinding_L = np.diag([lonA, lonI, lonA, lonI, lonA, lonI])
    Q_fixed_row2_L = np.concatenate([matBinding_L, Q_ligand], axis = 1)
    Q_fixed_L = np.concatenate([Q_fixed_row1, Q_fixed_row2_L], axis = 0)
    
    Q = Q_fixed_L - np.diag(np.sum(Q_fixed_L, axis=0))
    
    return Q

def p_ss_fn(param, mpmath_on = True):
    """
    Returns the steady-state probabilities of the 24 different states
    for the given choice of model parameters
    """
    
    # Calculate the transition rate matrix
    Q = Q_fn(param)
    
    # Find the steady-state vector
    if mpmath_on:
        # High precision version using the mpmath package (slow)
        # Solve the linear system of equations:
        # Qπ = 0 & 
        # sum(π_i) = 1
        
        Q[-1,:] = 1
        bnp = np.zeros(Q.shape[0])
        bnp[-1] = 1

        Q = mpmath.matrix(Q.tolist())
        b = bnp.tolist()
        return mpmath.lu_solve(Q,b)
    else:
        # Low precision version (sometimes erroneous) using the numpy package (fast)
        # Finds the nullspace of the Q-matrix
        # Normalizes the elements to sum to 1
        
        Q = np.array(Q, dtype = float)
        ns = sp.linalg.null_space(Q)
        ns = np.abs(ns[:,0])
        return ns/np.sum(ns)

def fidelity_abs_fn(param, mpmath_on=True, k_bT=mpf('1')):
    """
    Returns the absolute fidelity value, given the model parameters.
    
    The sets of indices for the right and wrong substrate-bound active states are:
    S_R = {4, 5}
    S_W = {9, 10}
    
    The absolute fidelity is defined as:
    η = Σ_{i∈S_R} π_i / Σ_{j∈S_W} π_j
    """
    
    # Calculate the steady-state probabilities
    [dG_1, dG_2, amplitude, alpha_21, alpha_54, alpha_32, alpha_65, loffA, loffI, kA, kI, kAL, kAS, lonI, lonA, gamma] = param

    p_ss = p_ss_fn(param, mpmath_on)
    
    # Find & return the absolute fidelity
    J_max_ss = gamma * (alpha_32 * np.exp((dG_1 + dG_2)/k_bT) * np.sqrt((alpha_21 * (alpha_32 + gamma)) / (alpha_32 * gamma))) \
    / (gamma * (2 * np.exp((dG_1 + dG_2) / (2 * k_bT)) + np.sqrt((alpha_21 * (alpha_32 + gamma)) / (alpha_32 * gamma))) \
    + alpha_32 * (2 * np.exp((dG_1 + dG_2) / (2 * k_bT)) + np.sqrt((alpha_21 * (alpha_32 + gamma)) / (alpha_32 * gamma)) \
    + np.exp((dG_1 + dG_2) / k_bT) * np.sqrt((alpha_21 * (alpha_32 + gamma)) / (alpha_32 * gamma))))

    J_obs = gamma * (p_ss[4] + p_ss[5] + p_ss[9] + p_ss[10])

    return float(J_max_ss/J_obs)

### ------- Parameter set generation ------- ###

# Number of values sampled for each rate parameter other than $r$
n_param = 4

# Number of sampled $r$ values
n_r = 20
# Lists of sampled rate values
dG_1_ls = [2]
dG_2_ls = [2]
amplitude_ls = np.linspace(-4, 4, n_param)
alpha_21_ls = [1]
alpha_54_ls = [1]
alpha_32_ls = [1]
alpha_65_ls = [1]
loffA_ls = np.logspace(-4, 4, n_param)
loffI_ls = np.logspace(-4, 4, n_param)
kA_ls = np.logspace(-4, 4, n_param)
kI_ls = np.logspace(-4, 4, n_param)
kAL_ls = np.logspace(-4, 4, n_param)
kAS_ls = np.logspace(-4, 4, n_param)
lonI_ls = np.logspace(-4, 4, n_param)
lonA_ls = np.logspace(-4, 4, n_param)
gamma_ls = np.logspace(-4, 4, n_r)

arguments = []
gamma_ls_all = []

counter = 0
for dG_1 in dG_1_ls:
    for dG_2 in dG_2_ls:
        for amplitude in amplitude_ls:
            for alpha_21 in alpha_21_ls:
                for alpha_54 in alpha_54_ls:
                    for alpha_32 in alpha_32_ls:
                        for alpha_65 in alpha_65_ls:
                            for loffA in loffA_ls:
                                for loffI in loffI_ls:
                                    for kA in kA_ls:
                                        for kI in kI_ls:
                                            for kAL in kAL_ls:
                                                for kAS in kAS_ls:
                                                    for lonI in lonI_ls:
                                                        for lonA in lonA_ls:
                                                            for gamma in gamma_ls:
                                                                param = [dG_1, dG_2, amplitude, alpha_21, alpha_54, alpha_32, alpha_65, loffA, loffI, kA, kI, kAL, kAS, lonI, lonA, gamma]
                                                                arguments.append((param, True))
                                                                gamma_ls_all.append(gamma)

# Run the simulation
fidelity_abs_ls = None
if __name__ == '__main__':
    with Pool(4) as p:
        fidelity_abs_ls = p.starmap(fidelity_abs_fn, arguments)
        

# Convert the lists into Numpy arrays
# fidelity_abs_ls = np.array(fidelity_abs_ls)
# gamma_ls_all = np.array(gamma_ls_all)

# # Unique $r$ values
# gamma_unique = np.unique(gamma_ls_all)

# # Number of significant digits kept for plottting data
# fidelity_abs_ls_sparse = []
# n_round = 3

# for i in range(len(gamma_unique)):
#     fidelity_gamma = np.sort(fidelity_abs_ls[gamma_ls_all == gamma_unique[i]])
#     fidelity_abs_ls_sparse.append(np.unique(np.round(fidelity_gamma,n_round)))
        
# with open('../resources/Fig_3D_data.pkl', 'wb') as f:
#     pickle.dump([fidelity_abs_ls_sparse, gamma_unique], f)