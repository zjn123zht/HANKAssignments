import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,r,w,vbeg_a_plus,vbeg_a,a,c,ell,l,u):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    # define post tax variables
    w_tilde = w*(1-par.tau_ell)
    r_tilde = r*(1-par.tau_a)

    # a. Loop over fixed states
    for i_fix in nb.prange(par.Nfix):
            
        # b. Loop over idiosyncratic state
        for i_z in nb.prange(par.Nz):

            # c. extract productivity and prepare
            z = par.z_grid[i_z] # productivity including zeta
            wt = z*w_tilde # income
            fac = (wt/par.varphi_grid[i_fix])**(1/par.nu) # factor for labour FOC

            # d. use FOCs for consumption and labour
            c_endo = (par.beta*vbeg_a_plus[i_fix,i_z,:])**(-1/par.sigma)
            ell_endo = fac*(c_endo)**(-par.sigma/par.nu)

            # e. define endogeneous and exogenous cash on hand grid
            m_endo = c_endo + par.a_grid - wt*ell_endo
            m_exo = (1+r_tilde)*par.a_grid # definition of exogenous cash on hand using the exogeneous asset grid

            # f. interpolation of c and ell to a common grid
            interp_1d_vec(m_endo,c_endo,m_exo,c[i_fix,i_z,:])
            interp_1d_vec(m_endo,ell_endo,m_exo,ell[i_fix,i_z,:])

            # g. Compute effective labour supply and exogenous assets given c and ell
            l[i_fix,i_z,:] = z*ell[i_fix,i_z,:]
            a[i_fix,i_z,:] = m_exo - c[i_fix,i_z,:] + wt*ell[i_fix,i_z,:]

            # h. Solve problem at the borrowing constraint
            for i_a in nb.prange(par.Na):
                
                # I. If borrowing constraint is violated
                if a[i_fix,i_z,i_a] < 0.0:

                    a[i_fix,i_z,i_a] = 0.0 # Set to borrowing constraint
                    
                    # II. solve FOC for ell
                    elli = ell[i_fix,i_z,i_a]

                    it = 0

                    # III. run Newton solver
                    while True:

                        # o. compute consumption and error in ell choice
                        ci = (1+r_tilde)*par.a_grid[i_a] + wt*elli
                        error = elli - fac*ci**(-par.sigma/par.nu)
                        
                        # oo. break if convergence, or compute nominator and denominator for Newton update 
                        if np.abs(error) < 1e-11:
                            break
                        else:
                            derror = 1 - fac*(-par.sigma/par.nu)*ci**(-par.sigma/par.nu-1)*wt
                            elli = elli - error/derror

                        it += 1
                        if it > par.max_iter_ell: raise ValueError('too many iterations')

                        # ooo. save optimal choices at the constraint
                        c[i_fix,i_z,i_a] = ci
                        ell[i_fix,i_z,i_a] = elli
                        l[i_fix,i_z,i_a] = z*elli

        
        # i. expectation step for continuation value in next iteration (previous period)
        v_a = (1+r_tilde)*c[i_fix,:,:]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a

        # j. compute instantaneous utility of choice 
        u[i_fix,:,:] = c[i_fix]**(1-par.sigma)/(1-par.sigma) - par.varphi_grid[i_fix]*ell[i_fix]**(1+par.nu) / (1+par.nu)
