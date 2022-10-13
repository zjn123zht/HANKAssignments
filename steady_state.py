import time
import numpy as np

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
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
    
    # a. varphi
    par.varphi_grid = np.array([par.varphi_min,par.varphi_max]) # Two possible states
    # par.varphi_grid = np.array([1.0])

    # b. a
    par.a_grid[:] = equilogspace(0.0,ss.w*par.a_max,par.Na)
    
    # c. z
    par.z_grid[:],z_trans,z_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz) # Transition probabilities do not depend on the fixed states per definition

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
        print(f' K_ss = {res.x[0]:8.4f}')
        print(f' L_ss = {res.x[1]:8.4f}')

    # Simulate the model at the found steady state once





































def find_ss_old(model,method='direct',do_print=False,K_min=1.0,K_max=4.0,NK=10):
    """ find steady state using the direct or indirect method """

    t0 = time.time()

    if method == 'direct':
        find_ss_direct(model,do_print=do_print,K_min=K_min,K_max=K_max,NK=NK)
    elif method == 'indirect':
        find_ss_indirect(model,do_print=do_print)
    else:
        raise NotImplementedError

    if do_print: print(f'found steady state in {elapsed(t0)}')

def find_ss_direct(model,do_print=False,K_min=1.0,K_max=10.0,NK=10):
    """ find steady state using direct method """

    # a. broad search
    if do_print: print(f'### step 1: broad search ###\n')

    K_ss_vec = np.linspace(K_min,K_max,NK) # trial values
    clearing_A = np.zeros(K_ss_vec.size) # asset market errors

    for i,K_ss in enumerate(K_ss_vec):
        
        try:
            clearing_A[i] = obj_ss(K_ss,model,do_print=do_print)
        except Exception as e:
            clearing_A[i] = np.nan
            print(f'{e}')
            
        if do_print: print(f'clearing_A = {clearing_A[i]:12.8f}\n')
            
    # b. determine search bracket
    if do_print: print(f'### step 2: determine search bracket ###\n')

    K_max = np.min(K_ss_vec[clearing_A < 0])
    K_min = np.max(K_ss_vec[clearing_A > 0])

    if do_print: print(f'K in [{K_min:12.8f},{K_max:12.8f}]\n')

    # c. search
    if do_print: print(f'### step 3: search ###\n')

    root_finding.brentq(
        obj_ss,K_min,K_max,args=(model,),do_print=do_print,
        varname='K_ss',funcname='A_hh-K'
    )

def find_ss_indirect(model,do_print=False):
    """ find steady state using indirect method """

    par = model.par
    ss = model.ss

    # a. exogenous and targets
    ss.L = 1.0
    ss.r = par.r_ss_target
    ss.w = par.w_ss_target

    assert (1+ss.r)*par.beta < 1.0, '(1+r)*beta < 1, otherwise problems might arise'

    # b. stock and capital stock from household behavior
    model.solve_hh_ss(do_print=do_print) # give us ss.a and ss.c (steady state policy functions)
    model.simulate_hh_ss(do_print=do_print) # give us ss.D (steady state distribution)
    if do_print: print('')

    ss.K = ss.A_hh = np.sum(ss.a*ss.D)
    
    # c. back technology and depreciation rate
    ss.Gamma = ss.w / ((1-par.alpha)*(ss.K/ss.L)**par.alpha)
    ss.rk = par.alpha*ss.Gamma*(ss.K/ss.L)**(par.alpha-1)
    par.delta = ss.rk - ss.r

    # d. remaining
    ss.Y = ss.Gamma*ss.K**par.alpha*ss.L**(1-par.alpha)
    ss.C = ss.Y - par.delta*ss.K
    ss.C_hh = np.sum(ss.D*ss.c)

    # e. print
    if do_print:

        print(f'Implied K = {ss.K:6.3f}')
        print(f'Implied Y = {ss.Y:6.3f}')
        print(f'Implied Gamma = {ss.Gamma:6.3f}')
        print(f'Implied delta = {par.delta:6.3f}') # check is positive
        print(f'Implied K/Y = {ss.K/ss.Y:6.3f}') 
        print(f'Discrepancy in K-A_hh = {ss.K-ss.A_hh:12.8f}') # = 0 by construction
        print(f'Discrepancy in C-C_hh = {ss.C-ss.C_hh:12.8f}\n') # != 0 due to numerical error 

def obj_ss_alt(r_ss,model,do_print=False):
    """ objective when solving for steady state capital """

    par = model.par
    ss = model.ss

    # a. set real interest rate and rental rate
    ss.r = r_ss
    ss.rk = ss.r+par.delta

    # b. production
    ss.Gamma = par.Gamma_ss # model user choice
    ss.K = (ss.rk/par.alpha*ss.Gamma)**(1/(par.alpha-1))
    ss.L = 1.0 # by assumption
    ss.Y = ss.Gamma*ss.K**par.alpha*ss.L**(1-par.alpha)    

    # b. implied wage
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

    if do_print: print(f'implied {ss.A_hh = :.4f}')

    # d. market clearing
    ss.clearing_A = ss.A_hh-ss.K

    ss.I = ss.K - (1-par.delta)*ss.K
    ss.C = ss.Y - ss.I
    ss.clearing_C = ss.C_hh-ss.C

    return ss.clearing_A # target to hit

