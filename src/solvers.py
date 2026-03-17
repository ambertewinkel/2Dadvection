import numpy as np
import logging
import warnings


def gcrk(A, b, x, kiter=10, jiter=5, tolerance=1e-6):
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
                logging.info(f"Converged at k,j: {k}, {j} with residual: {rmx}")

                return x

    logging.warning(f'GCR(k) did not converge within the given iterations (ktotal,jtotal={kiter},{jiter}). Final residual: {rmx}')
    warnings.warn(f'GCR(k) did not converge within the given iterations (ktotal,jtotal={kiter},{jiter}). Final residual: {rmx}')
    
    return x


def gmresm(A, b, x, kiter=10, jiter=5, tolerance=1e-6):
    """
    Matrixfree solution of linear Ax=b system using GMRES(m) method. (matrixfree through a function that computes Ax with def A(x))).
    However, GMRES(m) does need a small matrix H to be stored and solved (done here through np.linalg.solve). Apart from that, it currently stores a V matrix, arrays of size (m+1,N) where N is the size of the problem. This could be improved to reduce memory usage (memory usage is already improved with the restarting).
    --- IN --- 
    A: function to implement the A matrix
    b: N vector, rhs of equation
    x: N vector, initial guess for solution
    --- OUT ---
    x : converged (or cut short) solution
    """
    x = x.copy()

    r0 = b - A(x)

    reltol = tolerance * np.linalg.norm(b) # relative tolerance; see GMRES slides https://www.dmsa.unipd.it/~berga/Teaching/Phd/gmres_slides.pdf and Wikipedia https://en.wikipedia.org/wiki/Generalized_minimal_residual_method; I think MATLAB and Python compare the residual to the relative tolerance as well: https://www.mathworks.com/help/matlab/ref/gmres.html and https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html
    oldresidual = np.linalg.norm(r0)
    for irestart in range(kiter):
        h = np.zeros((jiter+1, jiter), dtype=np.float64)
        v = np.zeros((jiter+1, *np.shape(r0)), dtype=np.float64)
        v_hat = np.zeros(np.shape(r0), dtype=np.float64)
        v[0] = r0/np.linalg.norm(r0)

        for j in range(jiter):            
            Avj = A(v[j])
            for i in range(j+1):
                h[i, j] = np.dot(Avj.ravel(), v[i].ravel())

            v_hat = Avj.copy()
            for i in range(j+1):
                v_hat -= h[i, j] * v[i]

            h[j+1,j] = np.linalg.norm(v_hat)
            if h[j+1, j] < 1e-15:
                # exact Krylov solution found
                break
            
            v[j+1] = v_hat/h[j+1,j]
            
        e1 = np.zeros(jiter+1)
        e1[0] = oldresidual # np.linalg.norm(r0) # residual norm at start of restart loop
        y = np.linalg.lstsq(h, e1, rcond=None)[0] # residual here is the norm of the previous residual

        for i in range(len(y)):
            x += y[i] * v[i]

        r0 = b - A(x)
        residual = np.linalg.norm(r0)
        #residual_minimised = np.linalg.norm(e1 - h @ y)
     
        if residual < reltol: #tolerance: #residual_minimised < tolerance: 
            #print(f"Converged after restart {irestart} with residual {residual} (relative tolerance {reltol}).")
            break
        else:
            if residual > oldresidual: # if residual increased after restart, print a warning (this can happen with GMRES(m) if m is too small or the problem is hard)
                print(f"Residual increased after restart {irestart} (residual: {residual}, old: {oldresidual}). This may indicate a problem with the solver or the choice of parameters.")
        oldresidual = residual

    if residual >= reltol: 
        print(f'GMRES(m) did not converge within the given iterations (ktotal,jtotal={kiter},{jiter}). Final residual: {residual}, relative tolerance: {reltol}')

    return x