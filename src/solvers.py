import numpy as np


def gmresm(A, b, x, kiter=200, jiter=4, tolerance=1e-6, iterations_convergence=None, it=0):
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

    reltol = tolerance * np.linalg.norm(b) # relative tolerance

    norm_oldres = np.linalg.norm(r0)
    
    if norm_oldres < reltol:
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

            if residual < reltol and no_iters > 3:
                converged = True
                jend = j
                if iterations_convergence is not None:
                    iterations_convergence[it] = no_iters
                break
            else:
                if residual > norm_oldres: # if residual increased after restart, print a warning
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
        print(f'GMRES(m) did not converge within the given iterations (ktotal,jtotal={kiter},{jiter}). Final residual: {residual}, relative tolerance: {reltol}')

    return x
