import numpy as np
import logging
import warnings

def gcrk_matrixfree(config, A, b, x, kiter, jiter, tolerance=1e-6):
    """
    Matrixfree solution of linear Ax=b system using GCR(k) method. (matrixfree through a function that computes Ax with def A(x)))
    --- IN --- 
    A: function to implement the A matrix
    b: N vector, rhs of equation
    x: N vector, initial guess for solution
    --- OUT ---
    x : converged (or cut short) solution
    """
    x = x.copy()
    r = b - A(x)
    for k in range(kiter):
        p = np.zeros((jiter+1, *np.shape(r)))
        p[0] = r.copy() 
        for j in range(jiter):
            Ap = A(p[j])
            Ap2_sum = np.maximum(np.sum(Ap*Ap), 1e-15) # to avoid division by zero
            alpha = np.sum(r*Ap)/Ap2_sum
            x += alpha*p[j]
            r -= alpha*Ap
            beta = np.zeros(j+1)
            for i in range(j+1):
                Api = A(p[i])
                Api2_sum = np.maximum(np.sum(Api*Api), 1e-15)
                beta[i] = - np.sum(A(r)*Api)/Api2_sum
            p[j+1] = r + np.dot(np.rollaxis(p[0:j+1],0,3), beta)
            rmx = np.sqrt(np.max(r[:]*r[:])) # square root of max square residual
            if rmx < tolerance: 
                if config.verbose: logging.info(f"Converged at k,j: {k}, {j} with residual: {rmx}")
                return x

    logging.warning(f'GCR(k) did not converge within the given iterations (ktotal,jtotal={kiter},{jiter}). Final residual: {rmx}')
    warnings.warn(f'GCR(k) did not converge within the given iterations (ktotal,jtotal={kiter},{jiter}). Final residual: {rmx}')