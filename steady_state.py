import time
import numpy as np

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst, find_ergodic
from consav.misc import elapsed

import root_finding

# import scipy root finder
from scipy import optimize

def prepare_hh_ss(model):
    """ prepare the household block to solve for steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############
    
    # a. varphi and zeta
    par.varphi_grid = np.array([par.varphi_min,par.varphi_max]) # Two possible states
    par.zeta_grid = np.array([par.zeta_min,par.zeta_max])

    # b. a
    par.a_grid[:] = equilogspace(0.0,ss.w*par.a_max,par.Na)
    
    # c. z
    par.zt_grid[:],zt_trans,zt_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nzt) # Transition probabilities do not depend on the fixed states per definition

    # Combine z and zeta for productivity grid
    par.z_grid[:] = np.repeat(par.zeta_grid,par.Nzt)*np.tile(par.zt_grid,par.Nzeta)
    P_zeta = np.eye(par.Nzeta) # Nzeta x Nzeta identity matrix for fixed state
    z_trans = np.kron(P_zeta,zt_trans)
    z_trans_cumsum = np.cumsum(z_trans,axis=1)
    z_ergodic = np.tile(zt_ergodic,par.Nzeta)/par.Nzeta # This might be okay
    z_ergodic_cumsum = np.cumsum(z_ergodic)

    # print(np.sum(z_ergodic*par.z_grid)) # test if it sums to one

    #############################################
    # 2. transition matrix initial distribution #
    #############################################
    
    # Define the initial distribution of the transition matrix. Ensure sum of all elements is 1
    for i_varphi in range(par.Nvarphi):
        ss.z_trans[i_varphi,:,:] = z_trans # extract transition probabilities from defined markov chain
        ss.Dz[i_varphi,:] = z_ergodic / par.Nfix #  Divide by number of fixed states to ensure D sums to 1
        ss.Dbeg[i_varphi,:,0] = ss.Dz[i_varphi,:] # ergodic at a_lag = 0.0
        ss.Dbeg[i_varphi,:,1:] = 0.0 # none with a_lag > 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    # a. raw value
    y = ss.w*par.z_grid
    c = m = (1+ss.r)*par.a_grid[np.newaxis,:] + y[:,np.newaxis]
    v_a = (1+ss.r)*c**(-par.sigma)
    l = par.z_grid*1.0 # Effective labour supply

    # b. expectation
    ss.vbeg_a[:] = ss.z_trans@v_a

def obj_ss(x,model,do_print=False): # set x instead of K_ss
    """ objective when solving for steady state capital """
    print('it')
    K_ss = x[0]
    L_ss = x[1]

    par = model.par
    ss = model.ss

    # a. production
    ss.Gamma = par.Gamma_ss # model user choice
    ss.K = K_ss
    ss.L = L_ss #1.0225 #L_ss # set to L_ss instead
    ss.Y = ss.Gamma*ss.K**par.alpha*ss.L**(1-par.alpha)
    ss.G = par.G # extract government spending (constant), model user choice

    # b. implied prices
    ss.rk = par.alpha*ss.Gamma*(ss.K/ss.L)**(par.alpha-1.0)
    ss.r = ss.rk - par.delta
    ss.r_b = ss.r # from no arbitrage condition
    ss.w = (1.0-par.alpha)*ss.Gamma*(ss.K/ss.L)**par.alpha

    # c. household behavior
    if do_print:

        print(f'guess {ss.K = :.4f}')    
        print(f'implied {ss.r = :.4f}')
        print(f'implied {ss.w = :.4f}')

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    ss.A_hh = np.sum(ss.a*ss.D) # hint: is actually computed automatically
    ss.C_hh = np.sum(ss.c*ss.D)
    ss.L_hh = np.sum(ss.l*ss.D) # Aggregate effective labour supply

    taxes = par.tau_a*ss.r*np.sum(ss.a*ss.D) + par.tau_ell*np.sum(ss.w*ss.l*ss.D) # government tax income
    ss.B = (taxes - ss.G)/ss.r_b

    if do_print: print(f'implied {ss.A_hh = :.4f}')

    # d. market clearing
    ss.clearing_A = ss.A_hh-ss.K-ss.B
    ss.clearing_L = ss.L_hh-ss.L

    ss.I = ss.K - (1-par.delta)*ss.K # = delta*K
    ss.C = ss.Y - ss.I - ss.G # subtracted G for model to make sense
    ss.clearing_C = ss.C_hh-ss.C

    return np.array([ss.clearing_A,ss.clearing_L]) # target to hit


# find steady state function by mads
def find_ss(model,do_print=False): # add other inputs

    # Unpack parameters and ss variables
    par = model.par
    # ss = model.ss

    # Run root finder
    res = optimize.root(obj_ss,[3.15205947,1.0225],method='hybr',tol=par.tol_ss,args=(model))

    # print results
    if do_print:
        print('Solver terminated')
        # print(f' K_ss = {res.x[0]:8.4f}')
        # print(f' L_ss = {res.x[1]:8.4f}')

    # Simulate the model at the found steady state once
    # Print steady state aggregates and prices automatically (also consumption, bonds, investment, income and government spending, tax income)

    # # Run steady state

    obj_ss(res.x,model,do_print=False)

    if do_print:

        ss = model.ss
        print('')
        print('Steady state aggregates:')
        print(f' K_ss = {ss.K:8.4f}')
        print(f' L_ss = {ss.L:8.4f}')
        print(f' Y_ss = {ss.Y:8.4f}')
        print(f' G_ss = {ss.G:8.4f}')
        print(f' B_ss = {ss.B:8.4f}')
        print(f' I_ss = {ss.I:8.4f}')
        print(f' C_ss = {ss.C:8.4f}')
        print('')

        print('steady state prices:')
        print(f' w_ss = {ss.w:8.4f}')
        print(f' r_ss = {ss.r:8.4f}')
        print('')

        print('Check for market clearing:')
        print(f'Excess savings           ={ss.clearing_A:8.4f}')
        print(f'Exess consumption demand ={ss.clearing_C:8.4f}')
        print(f'Excess labour supply     ={ss.clearing_L:8.4f}')        


    # model.solve_hh_ss(do_print=do_print)
    # model.simulate_hh_ss(do_print=do_print)