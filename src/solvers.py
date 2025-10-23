import numpy as np

def gcrk_matrixfree(A, b, x, kiter, jiter, tolerance=1e-6): # Implemented with A matrixfree (i.e., a function that computes Ax through def A(x)))
    """
    Solving Ax=b with the GCR(k) algorithm.
    A: function to implement the A matrix
    b: N vector, rhs of equation
    x: N vector, initial guess for solution
    OUT: 
    p : converged (or cut short) solution
    """
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
            #if j == 0: # bit of a hacky solution to avoid the np dot error when beta is just a 1D vector.
            #    p[j+1] = r + np.sum(beta*p[0:j+1])
            #else:
            p[j+1] = r + np.dot(np.rollaxis(p[0:j+1],0,3), beta)#np.sum(np.dot(beta, p[0:j+1]))
            rmx = np.sqrt(np.max(r[:]*r[:])) # square root of max square residual
            if rmx < tolerance: 
                print("Converged at k,j:", k, j, "with residual:", rmx)
                return x
            #else:
            #    print(k+1, j+1, 'out of', kiter, jiter, '-> residual:', np.linalg.norm(r))

    print('Did not converge within the given iterations. Final residual:', rmx)
    return x
