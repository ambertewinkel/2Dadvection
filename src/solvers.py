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


def gmresm(A, b, x, kiter=10, jiter=5, tolerance=1e-6, irestarts_convergence=np.zeros(10), j_convergence=np.zeros(10), iterations_convergence=np.zeros(10), it=0):
    """
    Matrixfree solution of linear Ax=b system using GMRES(m) method. (matrixfree through a function that computes Ax with def A(x))).
    Semi-optimised version (i.e., implemented QR factorisation/least squares minimisation in Saad and Schultz 1986 p.860-862, but not the last step part).
    However, GMRES(m) does need a small matrix H (R_k here) to be stored and solved. Apart from that, it currently stores a V matrix, arrays of size (m+1,N) where N is the size of the problem. This could be improved to reduce memory usage (memory usage is already improved with the restarting).
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
    norm_oldres = np.linalg.norm(r0)
    
    

    #!!! diff
    if norm_oldres < reltol: # This could also be a problem. Now we are doing more iterations once the tolerance has not been achieved yet - but if it has been achieved, we don't do anything (which is fair) - so now you have a good number of time steps where you would maybe preferably want a better solution as well but you don't get this. Could this explain the difficulty in reducing the error for small max C (just beyond the 1.4 threshold?)
        print(f"Initial guess is already good enough with residual {norm_oldres} (relative tolerance {reltol}).")
        return x
    #reduction_tolerance = 1e-5*norm_oldres
    #print('values', norm_oldres, reltol, reduction_tolerance)
    converged = False

    no_iters = 0

    #jiter = 2
    for irestart in range(kiter):
        h = np.zeros((jiter+1, jiter), dtype=np.float64)
        cols = h.shape[1]
        v = np.zeros((jiter+1, *np.shape(r0)), dtype=np.float64)
        v_hat = np.zeros(np.shape(r0), dtype=np.float64)
        v[0] = r0/norm_oldres

        cos = np.zeros(jiter, dtype=np.float64)
        sin = np.zeros(jiter, dtype=np.float64)
        g = np.zeros(jiter+1, dtype=np.float64)
        g[0] = norm_oldres # residual norm at start of restart loop
        jend = jiter

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

            # Apply previous rotations
            for i in range(j):
                temp = cos[i]*h[i, j] - sin[i]*h[i+1, j]
                h[i+1, j] = sin[i]*h[i, j] + cos[i]*h[i+1, j]
                h[i, j] = temp

            # Compute new rotation
            denom = np.hypot(h[j+1, j], h[j, j]) # sqrt(h[j+1,j]*h[j+1,j] + h[j,j]*h[j,j])
            cos[j], sin[j] = h[j, j]/denom, - h[j+1, j]/denom

            # Rotating Hbar (builds R in Q Hbar = R factorisation)
            h[j, j], h[j+1, j] = cos[j]*h[j, j] - sin[j]*h[j+1, j], sin[j]*h[j, j] + cos[j]*h[j+1, j]

            # Apply rotation to g (effectively applying Q to g)
            temp = cos[j] * g[j] - sin[j] * g[j+1]
            g[j+1] = sin[j] * g[j] + cos[j] * g[j+1]
            g[j] = temp

            no_iters += 1





            #residual = abs(g[j+1])
            #norm_oldres = residual

            # Residual norm
            #print(g[j+1])
            #print()
            residual = abs(g[j+1])
            if residual < reltol and no_iters > 1:# and residual < reduction_tolerance: # and (j == jiter-1): # !!! check if I need to use norm(r0) or norm(b) for the relative tolerance... It might just be a choice.
                #print(f"Converged after restart {irestart, j} with residual {residual} (relative tolerance {reltol}).")
                converged = True
                jend = j
                #irestarts_convergence[it] = irestart
                #j_convergence[it] = j
                iterations_convergence[it] = no_iters #jiter * irestart + j + 1
                break
            else:
                if residual > norm_oldres: # if residual increased after restart, print a warning (this can happen with GMRES(m) if m is too small or the problem is hard)
                    raise ValueError(f"Residual increased after restart {irestart} (residual: {residual}, old: {norm_oldres}). This may indicate a problem with the solver or the choice of parameters.")
            norm_oldres = residual






        ######if residual < reltol:# and residual < reduction_tolerance: # and (j == jiter-1): # !!! check if I need to use norm(r0) or norm(b) for the relative tolerance... It might just be a choice.
            #print(f"Converged after restart {irestart, j} with residual {residual} (relative tolerance {reltol}).")
######            converged = True
######            jend = j
######            #irestarts_convergence[it] = irestart
######            #j_convergence[it] = j
######            iterations_convergence[it] = no_iters #jiter * irestart + j + 1
######
######        norm_oldres = residual
######
        #iterations_convergence[it] = no_iters #jiter * irestart + j + 1












        #y = np.linalg.lstsq(h[:cols, :cols], g[:cols], rcond=None)[0] # ! replace this (24-03-2026: I think h refers to R_k in the GMRES paper)

        # h (or R_k) is an upper triangular matrix. To find y, we need to minimise ||g-R_k y ||_2. I believe this means solving R_k y = g. --> We need to backsubstitute to find the y values, this should be easy because R_k is upper triangular.

        y = np.zeros(jiter)
        for ji in range(jend - 1, -1, -1):
            y[ji] = (g[ji] - np.dot(h[ji, ji + 1:], y[ji + 1:]))/h[ji, ji]

        for i in range(len(y)):
            x += y[i] * v[i]

        if converged:
            break

        r0 = b - A(x)

    if residual >= reltol: 
        print(f'GMRES(m) tryopt did not converge within the given iterations (ktotal,jtotal={kiter},{jiter}). Final residual: {residual}, relative tolerance: {reltol}')

    return x


def gmresm_old(A, b, x, kiter=10, jiter=5, tolerance=1e-6, irestarts_convergence=np.zeros(10), j_convergence=np.zeros(10), iterations_convergence=np.zeros(10), it=0):
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


    #!!! diff
    if oldresidual < reltol: # This could also be a problem. Now we are doing more iterations once the tolerance has not been achieved yet - but if it has been achieved, we don't do anything (which is fair) - so now you have a good number of time steps where you would maybe preferably want a better solution as well but you don't get this. Could this explain the difficulty in reducing the error for small max C (just beyond the 1.4 threshold?)
        print(f"Initial guess is already good enough with residual {oldresidual} (relative tolerance {reltol}).")
        return x
    #reduction_tolerance = 1e-5*oldresidual
    #print('values', oldresidual, reltol, reduction_tolerance)
    converged = False

    no_iters = 0

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
            no_iters += 1

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
            irestarts_convergence[it] = irestart
            #j_convergence[it] = j
            iterations_convergence[it] = no_iters
            break
        else:
            if residual > oldresidual: # if residual increased after restart, print a warning (this can happen with GMRES(m) if m is too small or the problem is hard)
                print(f"Residual increased after restart {irestart} (residual: {residual}, old: {oldresidual}). This may indicate a problem with the solver or the choice of parameters.")
        oldresidual = residual

    if residual >= reltol: 
        print(f'GMRES(m) did not converge within the given iterations (ktotal,jtotal={kiter},{jiter}). Final residual: {residual}, relative tolerance: {reltol}')

    #print(f'Number of iterations: {no_iters}')
    return x