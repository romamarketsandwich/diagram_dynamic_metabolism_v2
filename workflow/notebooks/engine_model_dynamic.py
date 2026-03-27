def Q_fn(param):
    """
    Returns the full 24x24 transition rate matrix of the coupled engine-enzyme system
    for the given set of model parameter
    """
    
    # To make sure that negative parameter values are not passed during numerical optimization 
    param_abs = []
    for el in param:
        param_abs.append(mpmath.fabs(el))
    
    # Unpack the parameters
    [konA, konI, loffA, loffI, kA, kI, kAL_option, kAS_option, lonI_H, lonA_H, f, koffR, koffW, r, kb, w] = param_abs
    
    delta = (lonA_H/lonI_H) * (loffI/loffA)
    if kAL_option == 0:
        kAL = kA
        kIL = kI / delta
    elif kAL_option == 1:
        kAL = np.sqrt(delta) * kA
        kIL = kI / np.sqrt(delta)
    else:
        kAL = delta*kA
        kIL = kI
        
    gamma = konA/konI
    if kAS_option == 0:
        kAS = kA
        kIS = kI / gamma
        kASL = kAL
        kISL = kIL / gamma
    elif kAS_option == 1:
        kAS = np.sqrt(gamma) * kA
        kIS = kI / np.sqrt(gamma)
        kASL = np.sqrt(gamma) * kAL
        kISL = kIL / np.sqrt(gamma)
    else:
        kAS = gamma*kA
        kIS = kI
        kASL = gamma*kAL
        kISL = kIL
    
    # Ligand binding rates in the compressed piston state
    lonI_L = lonI_H/f
    lonA_L = lonA_H/f
    
    #           ---------
    #           Q_fixed_L                    ------------
    #           ---------                    matExpansion
    #    Q_no_ligand   matUnbinding          ------------
    #    matBinding_L    Q_ligand
    #                                         ---------
    #          --------------                 Q_fixed_H
    #          matCompression                 ---------
    #          --------------           Q_no_ligand   matUnbinding
    #                                   matBinding_H    Q_ligand

    Q_no_ligand = np.array([[0, kA, koffR+r, 0, koffW+r, 0], \
                            [kI, 0, 0, koffR, 0, koffW],\
                            [konA, 0, 0, kAS, 0, 0], \
                            [0, konI, kIS, 0, 0, 0], \
                            [konA, 0, 0, 0, 0, kAS], \
                            [0, konI, 0, 0, kIS, 0]])
    
    Q_ligand = np.array([[0, kAL, koffR+r, 0, koffW+r, 0], \
                         [kIL, 0, 0, koffR, 0, koffW],\
                         [konA, 0, 0, kASL, 0, 0], \
                         [0, konI, kISL, 0, 0, 0], \
                         [konA, 0, 0, 0, 0 d, kASL], \
                         [0, konI, 0, 0, kISL, 0]])
    
    matUnbinding = np.diag([loffA, loffI, loffA, loffI, loffA, loffI])
    
    Q_fixed_row1 = np.concatenate([Q_no_ligand, matUnbinding], axis = 1)
    
    matBinding_L = np.diag([lonA_L, lonI_L, lonA_L, lonI_L, lonA_L, lonI_L])
    Q_fixed_row2_L = np.concatenate([matBinding_L, Q_ligand], axis = 1)
    Q_fixed_L = np.concatenate([Q_fixed_row1, Q_fixed_row2_L], axis = 0)
    
    matBinding_H = np.diag([lonA_H, lonI_H, lonA_H, lonI_H, lonA_H, lonI_H])
    Q_fixed_row2_H = np.concatenate([matBinding_H, Q_ligand], axis = 1)
    Q_fixed_H = np.concatenate([Q_fixed_row1, Q_fixed_row2_H], axis = 0)
    
    matExpansion = np.diag([kb+kb*(mpmath.exp(-w))*f]*6 + [kb+kb*(mpmath.exp(-w))]*6)
    matCompression = np.diag([kb+kb*(mpmath.exp(-w))/f]*6 + [kb+kb*(mpmath.exp(-w))]*6)
    
    Q_full_row1 = np.concatenate([Q_fixed_L, matExpansion], axis = 1)
    Q_full_row2 = np.concatenate([matCompression, Q_fixed_H], axis = 1)
    Q_full = np.concatenate([Q_full_row1, Q_full_row2], axis = 0)
    
    Q = Q_full - np.diag(np.sum(Q_full, axis=0))
    
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

def fidelity_abs_fn(param, mpmath_on = True):
    """
    Returns the absolute fidelity value, given the model parameters.
    
    The sets of indices for the right and wrong substrate-bound active states are:
    S_R = {2, 8}
    S_W = {4, 10}
    
    The absolute fidelity is defined as:
    η = Σ_{i∈S_R} π_i / Σ_{j∈S_W} π_j
    """
    
    # Calculate the steady-state probabilities
    p_ss = p_ss_fn(param, mpmath_on)
    
    # Find & return the absolute fidelity
    p_activeR = mpmath.fsum([p_ss[2], p_ss[8]])
    p_activeW = mpmath.fsum([p_ss[4], p_ss[10]])
    return float(p_activeR/p_activeW)