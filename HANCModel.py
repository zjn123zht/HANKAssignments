import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass

import steady_state
import household_problem

class HANCModelClass(EconModelClass,GEModelClass):    

    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par','ini','ss','path','sim'] # not used today: 'ini', 'path', 'sim'

        # not used today: .sim and .path
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['r','w'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs (not used today)
        self.outputs_hh = ['a','c','ell','l'] # outputs, added 'l' for effective labour
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = [] # exogenous shocks (not used today)
        self.unknowns = [] # endogenous unknowns (not used today)
        self.targets = [] # targets = 0 (not used today)

        # d. all variables
        self.varlist = [
            'Y','C','I','Gamma','K','L','G','B',
            'rk','w','r','r_b',
            'A_hh','C_hh',
            'clearing_A','clearing_C', 'clearing_L']

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        self.block_pre = None # not used today
        self.block_post = None # not used today

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 2 # number of fixed discrete states, two values of varphi
        par.Nz = 7 # number of stochastic discrete states (here productivity)

        # a. preferences
        par.beta = 0.96 # discount factor
        par.sigma = 2.0 # CRRA coefficient
        par.varphi_min = .9
        par.varphi_max = 1.1
        # par.varphi = 1.0
        par.zeta = 1.0 # fixed individual productivity component
        par.nu = 1.0

        # b. income parameters
        par.rho_z = 0.96 # AR(1) parameter
        par.sigma_psi = 0.10 # std. of shock

        # Government stuff
        par.tau_a = .1
        par.tau_ell = .3
        par.G = .3

        # c. production and investment
        par.alpha = 0.3 # cobb-douglas
        par.delta = 0.1 # depreciation rate
        par.Gamma_ss = 1.0 # direct approach: technology level in steady state

        # f. grids         
        par.a_max = 100.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # g. indirect approach: targets for stationary equilibrium
        par.r_ss_target = 0.03
        par.w_ss_target = 1.0

        # h. misc.
        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        par.max_iter_ell = 30

        par.tol_solve = 1e-12 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem
        par.tol_ss = 1e-8 # tolerance when finding steady state

        
    def allocate(self):
        """ allocate model """

        par = self.par

        # a. grids
        par.Nvarphi = par.Nfix
        par.varphi_grid = np.zeros(par.Nvarphi)
        
        # b. solution
        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss
