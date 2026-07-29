import numpy as np
import logging
import warnings


def gcrk(A, b, x, kiter=10, jiter=4, tolerance=1e-6):
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


def gmresm_opt(A, b, x, kiter=10, jiter=4, tolerance=1e-6, iterations_convergence=np.zeros(10), it=0):
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
    
    if norm_oldres < reltol:
        print(f"Initial guess is already good enough with residual {norm_oldres} (relative tolerance {reltol}).")
        return x
    converged = False

    no_iters = 0

    for irestart in range(kiter):
        h = np.zeros((jiter+1, jiter), dtype=np.float64)
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
            residual = abs(g[j+1])
            if residual < reltol:# and no_iters > 3: #no_iters > 1:
                #print(f"Converged after restart {irestart, j} with residual {residual} (relative tolerance {reltol}).")
                converged = True
                jend = j
                iterations_convergence[it] = no_iters #jiter * irestart + j + 1
                break
            else:
                if residual > norm_oldres: # if residual increased after restart, print a warning (this can happen with GMRES(m) if m is too small or the problem is hard)
                    raise ValueError(f"Residual increased after restart {irestart} (residual: {residual}, old: {norm_oldres}). This may indicate a problem with the solver or the choice of parameters.")
            norm_oldres = residual

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


def gmresm(A, b, x, kiter=10, jiter=4, tolerance=1e-6, iterations_convergence=np.zeros(10), it=0):
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

    if oldresidual < reltol: # Was not originally in the old GMRES(m)
        print(f"Initial guess is already good enough with residual {oldresidual} (relative tolerance {reltol}).")
        return x

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
            #print(no_iters, irestart)

        e1 = np.zeros(jiter+1)
        e1[0] = oldresidual # np.linalg.norm(r0) # residual norm at start of restart loop
        y = np.linalg.lstsq(h, e1, rcond=None)[0] # residual here is the norm of the previous residual

        for i in range(len(y)):
            x += y[i] * v[i]

        r0 = b - A(x)

        residual = np.linalg.norm(r0)
        #residual_minimised = np.linalg.norm(e1 - h @ y)
        maxr0 = np.max(np.abs(r0)) # = maxr0
        #residual = maxr0

        if residual < reltol or maxr0 < reltol/np.sqrt(b.size): #tolerance: #residual_minimised < tolerance: 
            #print(f"Converged after restart {irestart} with residual {residual} (relative tolerance {reltol}).")
            iterations_convergence[it] = no_iters
            break
        else:
            if residual > oldresidual: # if residual increased after restart, print a warning (this can happen with GMRES(m) if m is too small or the problem is hard)
                print(f"Residual increased after restart {irestart} (residual: {residual}, old: {oldresidual}). This may indicate a problem with the solver or the choice of parameters.")
        oldresidual = residual

    if residual >= reltol and maxr0 >= reltol/np.sqrt(b.size): # This second part wasn't here yet for the trial run.
        print(f'GMRES(m) did not converge within the given iterations (ktotal,jtotal={kiter},{jiter}). Final residual: {residual}, relative tolerance: {reltol}')

    #print(f'Number of iterations: {no_iters}')
    return x