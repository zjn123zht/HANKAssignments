import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,r,w,vbeg_a_plus,vbeg_a,a,c,ell,l):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    # Loop over fixed states
    for i_fix in nb.prange(par.Nfix):
            
        # a. solve step
        for i_z in nb.prange(par.Nz):
            # print(vbeg_a_plus[i_fix,i_z,:])
            # define post tax variables
            w_tilde = w*(1-par.tau_ell)
            r_tilde = r*(1-par.tau_a)

            # prepare
            z = par.z_grid[i_z]
            wt = z*w_tilde # use a list for zeta in the future as it is a state
            fac = (wt/par.varphi_grid[i_fix])**(1/par.nu)

            # use FOCs
            c_endo = (par.beta*vbeg_a_plus[i_fix,i_z,:])**(-1/par.sigma) # parallel over asset states
            ell_endo = fac*(c_endo)**(-par.sigma/par.nu) # Find labour choice
            # l_endo = par.zeta*z*ell_endo # effective labour supply, not needed
            # print(c_endo)
            # print(ell_endo)
            # interpolation of c and ell to a common grid
            m_endo = c_endo + par.a_grid - wt*ell_endo
            m_exo = (1+r_tilde)*par.a_grid # definition of exogenous cash on hand using the exogeneous asset grid
            
            interp_1d_vec(m_endo,c_endo,m_exo,c[i_fix,i_z,:])
            interp_1d_vec(m_endo,ell_endo,m_exo,ell[i_fix,i_z,:])
            l[i_fix,i_z,:] = par.zeta*z*ell[i_fix,i_z,:] # effective labour supply over an exogeneous grid

            a[i_fix,i_z,:] = m_exo - c[i_fix,i_z,:] + wt*ell[i_fix,i_z,:]

            # Refinement at borrowing constraint
            for i_a in range(par.Na):
                
                # If borrowing constraint is violated
                if a[i_fix,i_z,i_a] < 0.0:

                    a[i_fix,i_z,i_a] = 0.0 # Set to borrowing constraint
                    
                    # Solve FOC for ell
                    elli = ell[i_fix,i_z,i_a]

                    it = 0
                    while True:

                        ci = (1+r_tilde)*par.a_grid[i_a] + wt*elli
                        error = elli - fac*ci**(-par.sigma/par.nu)
                        if np.abs(error) < 1e-11:
                            break
                        else:
                            derror = 1 - fac*(-par.sigma/par.nu)*ci**(-par.sigma/par.nu-1)*wt # denominator for Newton step
                            elli = elli - error/derror

                        it += 1
                        if it > par.max_iter_ell: raise ValueError('too many iterations')

                        # Save choices
                        c[i_fix,i_z,i_a] = ci
                        ell[i_fix,i_z,i_a] = elli
                        l[i_fix,i_z,i_a] = par.zeta*z*elli
        
        # b. expectation step
        v_a = (1+r_tilde)*c[i_fix,:,:]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a