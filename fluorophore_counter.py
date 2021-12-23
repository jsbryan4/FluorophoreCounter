
import time
import copy
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy import stats
from matplotlib import gridspec
from joblib import Parallel, delayed
from algorithms import fast_digamma, HistoryH5, Dirichlet, FFBS
from scipy.special import gammaln

PARAMETERS = {

    # constants
    'dt': 1,        # time step in seconds
    'gain': None,   # gain                  <<REQUIRED>>

    # variables
    'states': None,       # photostates
    'num_flor': None,     # number of fluorophores present
    'mu_flor': None,      # brightness of fluorophore
    'mu_back': None,      # brightness of background
    'transitions': None,  # photostate transition probability matrix
    'P': None,            # probability

    # priors
    'load_weight': .1,         # prior on load
    'mu_flor_mean': None,      # prior mean on fluorophore brightness
    'mu_flor_shape': 100,      # prior shape on fluorophore brightness
    'mu_back_mean': None,      # prior mean on background brightness
    'mu_back_shape': 10,       # prior shape of background brightness
    'transitions_conc': None,  # prior concentration on photostate transition matrix

    # numbers
    'num_rois': None,    # number of ROIs
    'num_data': None,    # number of time levels
    'num_states': 3,     # number of states
    'num_load': None,    # number of loads
    'num_end': None,     # number of frames we force states to be photobleached

    # sampler parameters
    'seed': None,                   # RNG seed
    'flor_brightness_guess': None,  # guess for fluorophore brightness <<REQUIRED>>
    # 'eps': .1,                      # numerical stability parameter
    'num_iter': 10000,              # number of Gibbs sample iterations
    'num_together': 2,              # number of phototrajectories sampled together
    'state_combos': None,           # joint photostate combinations
    'degenerate_combos': None,      # degenerate joint-photostate combos
    'num_combos': None,             # number of joint-photostate combinations
    'parpool_status': False,        # if true then use parallel computing
}


@nb.njit(cache=True)
def states_to_pops(states, num_states):
    num_data = states.shape[1]
    pops = np.zeros((num_states, num_data))
    for k in range(num_states):
        pops[k, :] = np.sum(states == k, axis=0)
    return pops


class FluorophoreCounter:
    """
    This is the fluorophore counter class.
    """

    def __init__(self):

        self.data = None
        self.history = None

        return

    @staticmethod
    def simulate_data(parameters=None, **kwargs):
        """
        This code generates simulated data using the forward model.

        :param parameters: Parameters can be specified by passing a
        dictionary filled with variables to be used in the simulation
        such as
        simulate_data(parameters={'num_states': 4, 'mu_flor': 50})
        :param kwargs: Alternatively, parameters of the simulation
        can be specified one by one using key word arguments such as
        simulate_data(num_states=4, mu_flor=50)
        :return: This function returns simulated data as well as a
        dictionary containing the ground truth values from the
        simulation.
        """

        # merge default parameters with input parameters
        default_parameters = {

            # simulation parameters
            'seed': 0,           # seed for random number generator
            'num_sites':  20,    # number of binding sites
            'binding_eff':  .7,  # binding efficiency

            # constants
            'dt': 1,        # time step
            'gain': 20,   # gain

            # variables
            'states': None,       # photostates
            'num_flor': None,     # number of fluorophores present
            'mu_flor': 100,      # brightness of fluorophore
            'mu_back': 100,      # brightness of background
            'transitions': None,  # photostate transition probability matrix

            # numbers
            'num_rois': 1,      # number of ROIs
            'num_data': 2000,    # number of time levels
            'num_states': 3,     # number of states

        }
        if parameters is None:
            parameters = {**default_parameters, **kwargs}
        else:
            parameters = {**default_parameters, **parameters, **kwargs}

        # set RNG seed
        np.random.seed(parameters['seed'])

        # extract parameters
        gain = parameters['gain']
        eff = parameters['binding_eff']
        mu_flor = parameters['mu_flor']
        mu_back = parameters['mu_back']
        pi = parameters['transitions']
        num_rois = parameters['num_rois']
        num_data = parameters['num_data']
        num_sites = parameters['num_sites']
        num_states = parameters['num_states']
        if pi is None:
            pi = np.ones((num_states + 1, num_states))
            # increase self transition
            pi[:-1, :] += 100 * np.eye(num_states)
            # photobleached only self transitions
            pi[-2, :-1] = 0
            # fluorophores cannot start photobleached
            pi[-1, -1] = 0
            # no blink to bleached transitions
            if num_states > 2:
                pi[0, -1] = 0
            for k in range(num_states + 1):
                pi[k, :] /= np.sum(pi[k, :])
            # create effective pi by allowing nonbinding flurorophores to start bleached
            pi[-1, :] = np.array([*((1 - eff) * pi[-1, :-1]), eff])
            parameters['transitions'] = pi
        if np.isscalar(mu_flor):
            if num_states == 2:
                mu_flor = np.array([1, 0]) * mu_flor
            else:
                mu_flor = np.array([0, 1, *np.random.rand(num_states - 3), 0]) * mu_flor
            parameters['mu_flor'] = mu_flor
        if np.isscalar(mu_back):
            mu_back = np.ones(num_rois) * mu_back
            parameters['mu_back'] = mu_back
        # make effective transition matrix
        pi_eff = np.zeros((num_states + 1, num_states))
        pi_eff[:-1, :] = pi[:-1, :]
        pi_eff[-1, :] = pi[-1, :] * eff + (1 - eff) * np.array([0] * (num_states - 1) + [1])

        # simulate data
        """
        We wrap the HMM sampler in a numba jit wrapper for faster
        sampling of states.
        """
        @nb.njit(cache=True)
        def sample_HMM_nb(transition_matrix, num_data):

            # set variables
            num_states = transition_matrix.shape[1]
            states = np.zeros(num_data, dtype=np.int64)

            # normalize the transition matrix
            wp = transition_matrix.copy()
            for k in range(num_states + 1):
                wp[k, :] = np.cumsum(wp[k, :])

            # sample initial state
            states[0] = np.searchsorted(wp[-1, :], np.random.rand())
            for n in range(1, num_data):
                # sample each following state from the row of the transition matrix corresponding to the previous state
                states[n] = np.searchsorted(wp[states[n - 1], :], np.random.rand())

            return states

        data = np.zeros((num_rois, num_data))
        states = np.zeros((num_rois, num_sites, num_data), dtype=int)
        num_flor = np.zeros(num_rois, dtype=int)
        for r in range(num_rois):
            # find the state of each fluorophore at each time
            for l in range(num_sites):
                # sample whether or not the fluorophore is bound
                while True:
                    # sample states within a while loop to ensure that they all end up photobleached
                    states[r, l, :] = sample_HMM_nb(pi_eff, num_data)
                    if states[r, l, -1] == num_states - 1:
                        break

            # total number of fluorophores is the number that do not start photobleached
            num_flor[r] = np.sum(states[r, :, 0] < num_states - 1)

            # sample the brightness data
            brightness = mu_flor @ states_to_pops(states[r, :, :], num_states) + mu_back[r]
            data[r, :] = stats.gamma.rvs(a=brightness, scale=gain)

        parameters['states'] = states
        parameters['num_flor'] = num_flor

        return data, parameters

    def initialize_variables(self, data, parameters) -> object:
        """
        This script initializes the Gibbs sampler values
        using the input data and chosen parameters.

        :param data: This is the data which is being analyzed.
        :param parameters: These are the parameters chosen
        for inference.
        :return: This script returns a Simplenamespace object
        containing a set of variables.
        """

        val = SimpleNamespace(**parameters)

        # extract parameters
        flor_brightness_guess = val.flor_brightness_guess
        gain = val.gain
        mu_flor_mean = val.mu_flor_mean
        mu_flor_shape = val.mu_flor_shape
        mu_back_mean = val.mu_back_mean
        mu_back_shape = val.mu_back_shape
        transitions_conc = val.transitions_conc
        load_weight = val.load_weight
        num_end = val.num_end
        num_load = val.num_load
        num_states = val.num_states
        num_together = val.num_together

        # set up numbers
        num_rois, num_data = data.shape
        # use ruler method to set initial guess for the number of fluorophores
        num_estimated = np.max(np.max(data, axis=1) - np.min(data, axis=1)) / flor_brightness_guess
        if num_end is None:
            # assume last one percent of the data is photobleached
            num_end = int(round(num_data / 100))
        if num_load is None:
            num_load = int(round(1.5 * num_estimated, -1))

        # set up priors
        if num_states == 2:
            bright_idx = np.array([1, 0])
        else:
            bright_idx = np.array([0, *np.ones(num_states - 2), 0])
        if mu_flor_mean is None:
            mu_flor_mean = bright_idx * flor_brightness_guess / gain
        if np.isscalar(mu_flor_shape):
            # place a sharp prior on fluorophore brightness
            mu_flor_shape = bright_idx * mu_flor_shape
        if mu_back_mean is None:
            # last data point is the estimate for background
            mu_back_mean = np.mean(data[:, -num_end:], axis=1) / gain
        if np.isscalar(mu_back_shape):
            # place a sharp prior on background
            mu_back_shape = np.ones(num_rois) * mu_back_shape
        if transitions_conc is None:
            pi = np.ones((num_states + 1, num_states))
            # fluorophores cannot start photobleached
            pi[-1, -1] = 0
            # increase self transition
            pi[:-1, :] += 100 * np.eye(num_states)
            # photobleached can only self transition
            pi[-2, :-1] = 0
            # do not allow dark to bleached transitions
            if num_states > 2:
                pi[0, -1] = 0
            # normalize the matrix
            for k in range(num_states + 1):
                pi[k, :] /= np.sum(pi[k, :])
            transitions_conc = pi

        # calculate joint photostate combinations
        """
        The joint photostate combinations are needed to sample
        multiple phototrajectories simultaneously. That is,
        instead of sampling the phototrajectory of fluorophore A
        and then sampling the phototrajectory of fluorophore B,
        we sample the joint phototrajectory of fluorophore A and
        B in order to get better mixing. If, for example, each
        fluorphore can either be bright or dark, then the joint
        state space would have four components:
        1) A bright and B bright
        2) A bright and B dark
        3) A dark and B bright
        4) A dark and B dark
        Here we generalize to include multiple states per
        fluorophore and more than 2 fluorophores per joint
        state space.
        """
        num_combos = num_states ** num_together
        state_combos = np.zeros((num_combos, num_together), dtype=int)
        for k in range(num_together):
            state_combos[:, k] = np.tile(
                np.repeat(np.arange(num_states), num_states ** (num_together - k - 1)),
                num_states ** k
            )
        # reduce number of joint photostate combinations by equating states with equal brightness
        """
        Note that many of the states in the joint state space
        will have the same brightness. For example, A dark with
        B brigth will be the same brightness as A bright with B
        dark. Therefore, to speed up calculations, instead of 
        caclulating brightness of A-bright-B-dark and A-dark-B-bright
        seperately, we can calculate the brightness of
        one-fluorophore-bright and use the value for both of the
        above joint states.
        """
        degenerate_combos = state_combos.copy()
        for k in np.where(val.mu_flor_mean == 0)[0]:
            degenerate_combos[degenerate_combos == k] = num_states - 1
        degenerate_combos = np.sort(degenerate_combos, axis=1)

        # load val with calculated values
        val.num_data = num_data
        val.num_rois = num_rois
        val.num_end = num_end
        val.num_load = num_load
        val.mu_flor_mean = mu_flor_mean
        val.mu_flor_shape = mu_flor_shape
        val.mu_back_mean = mu_back_mean
        val.mu_back_shape = mu_back_shape
        val.transitions_conc = transitions_conc
        val.load_weight = load_weight
        val.num_combos = num_combos
        val.state_combos = state_combos
        val.degenerate_combos = degenerate_combos

        # set up variables
        states = (num_states - 1) * np.ones((num_rois, num_load, num_data), dtype=int)
        num_flor = np.zeros(num_rois)
        mu_flor = stats.norm.rvs(mu_flor_mean, .01) * (mu_flor_mean > 0)
        mu_back = stats.norm.rvs(mu_back_mean, .01)
        transitions = Dirichlet.sample(1000 * transitions_conc)
        val.states = states
        val.num_flor = num_flor
        val.mu_flor = mu_flor
        val.mu_back = mu_back
        val.transitions = transitions
        val.P = self.posterior(val)

        return val

    def posterior(self, val, **kwargs) -> float:
        """
        This functions calculates the value of the un-normalized
        log pdf for a given set of random variables.

        :param val: This parameter is a Simplenamespace object
        containing the stored value of each random variable
        and parameter in the posterior.
        :param kwargs: Any parameters specified in keyword arguments
        will be override those stored in val.
        :return: This functions outputs the un-normalized log posterior
        calculated with the given values.
        """

        data = self.data

        # override val with parameters specified via kwargs
        val = copy.deepcopy(val)
        for key, value in kwargs.items():
            setattr(val, key, value)

        # extract parameters
        gain = val.gain
        states = val.states
        pi = val.transitions
        pi_conc = val.transitions_conc
        mu_flor = val.mu_flor
        mu_flor_mean = val.mu_flor_mean
        mu_flor_shape = val.mu_flor_shape
        mu_back = val.mu_back
        mu_back_mean = val.mu_back_mean
        mu_back_shape = val.mu_back_shape
        load_weight = val.load_weight
        num_rois = val.num_rois
        num_load = val.num_load
        num_data = val.num_data
        num_states = val.num_states

        # calculate shape parameters
        idx = mu_flor_mean > 0
        mu_flor_scale = np.zeros(mu_flor_mean.shape)
        mu_flor_scale[idx] = mu_flor_mean[idx] / mu_flor_shape[idx]
        mu_back_scale = mu_back_mean / mu_back_shape
        # calculate effective pi for collapsed state space when weight on load is taken into account
        pi_eff = pi.copy()
        pi_eff[-1, :] *= load_weight
        pi_eff[-1, -1] = 1 - load_weight

        # probability from likelihood
        brightness = np.zeros(shape=data.shape)
        for r in range(num_rois):
            brightness[r, :] = mu_flor @ states_to_pops(states[r, :, :], num_states) + mu_back[r]
        lhood = np.sum(stats.gamma.logpdf(data, a=brightness, scale=gain))

        # probability from phototrajectory
        kinetic = 0
        for i in range(num_states):
            if pi_eff[-1, i] > 0:
                kinetic += np.sum(states[:, :, 0] == i) * np.log(pi_eff[-1, i])
            for j in range(num_states):
                if pi_eff[i, j] > 0:
                    kinetic += np.sum((states[:, :, :-1] == i) * (states[:, :, 1:] == j)) * np.log(pi_eff[i, j])

        # probability from prior
        prior = (
            # prior on fluorophore brightness (ignore dark states)
            np.sum(stats.gamma.logpdf(mu_flor[idx], a=mu_flor_shape[idx], scale=mu_flor_scale[idx]))
            # prior on background brightness
            + np.sum(stats.gamma.logpdf(mu_back, a=mu_back_shape, scale=mu_back_scale))
            # prior on transitions
            + np.sum(Dirichlet.logpdf(pi, pi_conc))
        )

        prob = lhood + kinetic + prior

        return prob

    def sample_mu(self, val) -> None:
        """
        This function samples the brightness parameters,
        mu_flor and mu_back using Hamiltonian Monte Carlo.

        :param val: This parameter is a Simplenamespace object
        containing the stored value of each random variable
        and parameter in the posterior.
        :return: This function returns val, but with an
        updated value for mu_flor and mu_back.
        """

        # get data
        data = self.data.reshape((1, -1))

        # get values
        gain = val.gain
        states = val.states
        mu_flor = val.mu_flor
        mu_flor_mean = val.mu_flor_mean
        mu_flor_shape = val.mu_flor_shape
        mu_back = val.mu_back
        mu_back_mean = val.mu_back_mean
        mu_back_shape = val.mu_back_shape
        num_data = val.num_data
        num_rois = val.num_rois
        num_states = val.num_states

        # initialze variables
        num_vars = num_states + num_rois
        idx = np.where(val.mu_flor_mean > 0)[0]
        # shape
        shape = np.zeros((num_vars, 1))
        shape[:num_states, 0] = mu_flor_shape[:]
        shape[num_states:, 0] = mu_back_shape
        # scale
        scale = np.zeros((num_vars, 1))
        scale[idx, 0] = mu_flor_mean[idx] / mu_flor_shape[idx]
        scale[num_states:, 0] = (mu_back_mean / mu_back_shape)[:]

        # initialize a mu vector containing the variables we wish to sample, mu_flor and mu_back
        q = np.zeros((num_vars, 1))
        q[:num_states, 0] = mu_flor[:]
        q[num_states:, 0] = mu_back[:]
        q_old = q.copy()
        idy = q > 0  # keep track of which states are dark (we only sample bright states)
        num_var = q.shape[0]

        # hmc dynamics variables
        h = np.random.exponential() / 100
        masses = (1 + np.random.pareto(1, size=q.shape))
        masses_inv = np.zeros(shape=masses.shape)  # negative mass is interpretted as an unchanging variable
        masses_inv[masses > 0] = 1 / masses[masses > 0]
        num_steps = np.random.poisson(25)

        # create populations array
        pops = np.zeros((num_vars, num_rois * num_data))
        """
        pops is an array such that each element i, j corresponds to the 
        multiplicitive factor in front of q[i] for data point j in the 
        likelihood. For example, if in ROI 1 at time level 17 there are two
        fluorophores in the bright state, then we find the element, j,
        corresponding to ROI 1 and time level 17, and we find the element,
        i, corresponding to the bright state, and we set q[i,j]=2 (because
        there are two bright fluorophores), then we would find the i
        corresponding to the background brightness of ROI 1, and we would
        set this q[i,j]=1 (the multiplicitive factor in front of the 
        background brightness is 1 when it is the corresponding ROI and 0
        otherwise).
        """
        for r in range(num_rois):
            idx = np.arange(r*num_data, (r+1)*num_data)
            pops[:num_states, idx] = states_to_pops(states[r, :, :], num_states)
            pops[num_states + r, idx] = 1

        # the conditional probability for the mu vector
        def probability(q_, p_):
            if np.sum(q_ < 0) > 0:
                prob = -np.inf
            else:
                prob = (
                    np.sum(stats.gamma.logpdf(data, a=q_.T @ pops, scale=gain))  # likelihood
                    + np.sum(stats.gamma.logpdf(q_[idy], a=shape[idy], scale=scale[idy]))  # prior
                    + np.sum(stats.norm.logpdf(p_[idy], loc=0, scale=np.sqrt(masses[idy])))  # momentum
                )
            return prob

        # the gradient of the Hamiltonian with respect to the mu_vector
        def dH_dq(q_):
            if np.any(q_ < 0):
                """
                In the event that q_new becomes negative, fast_digamma becomes
                slow. Since q should never be negative anyway, there is no
                need for further computation and we can skip this step knowing
                that this value of q will be rejected anyway.
                """
                return q_
            q_new = np.zeros(q_.shape)
            q_new[idy] = (
                    (shape[idy] - 1) / q_[idy] - 1 / scale[idy]
                    + (pops @ (np.log(data / gain) - fast_digamma(q_.T @ pops)).T)[idy]
            )
            return q_new

        # sample momentum
        p = np.random.randn(num_var, 1) * np.sqrt(masses)
        p_old = p.copy()

        # run the HMC
        for i in range(num_steps):
            p = p + .5 * h * dH_dq(q)
            q = q + h * p * masses_inv
            p = p + .5 * h * dH_dq(q)

        # find acceptance ratio
        P_new = probability(q, p)
        P_old = probability(q_old, p_old)
        if (P_new - P_old) < np.log(np.random.rand()):
            q = q_old

        # update the new mu values
        val.mu_flor[:] = q[:num_states, 0]
        val.mu_back[:] = q[num_states:, 0]

        return

    def sample_states(self, val) -> None:
        """
        This function samples the phototrajectory of each fluorophore.

        :param val: This parameter is a Simplenamespace object
        containing the stored value of each random variable
        and parameter in the posterior.
        :return: This function returns val, but with an
        updated phototrajectory.
        """

        # get data
        data = self.data

        # get values
        pi = val.transitions
        gain = val.gain
        states = val.states
        mu_flor = val.mu_flor
        mu_back = val.mu_back
        load_weight = val.load_weight
        num_end = val.num_end
        num_data = val.num_data
        num_load = val.num_load
        num_rois = val.num_rois
        num_states = val.num_states
        num_combos = val.num_combos
        num_together = val.num_together
        degenerate_combos = val.degenerate_combos
        state_combos = val.state_combos
        parpool_status = val.parpool_status

        # calculate effective transition matrix for combined state space
        """
        We create an effective state space in which "load off" fluorophores
        are the same state as "load on photobleached" fluorophores. This 
        allows us to sample the load and the phototrajectory at the same time.
        """
        pi_comb = np.zeros((num_combos + 1, num_combos))
        pi_comb_body = pi[:-1, :].copy()
        pi_comb_init = np.array([*pi[-1, :-1] * load_weight, 1 - load_weight])
        """
        Here we use our effective transition matrix to calculate the transition
        probability matrix of the joint state space. The joint state space
        transition matrix is the kronecker product of the effective state space
        matrix.
        """
        for n in range(1, num_together):
            pi_comb_body = np.kron(pi_comb_body, pi[:-1, :])
            pi_comb_init = np.kron(pi_comb_init, np.array([*pi[-1, :-1] * load_weight, 1 - load_weight]))
        pi_comb[:-1, :] = pi_comb_body
        pi_comb[-1, :] = pi_comb_init

        # sample trajectory
        def sample_states_r(r):
            """
            This function samples the phototrajectory for ROI r.
            """

            print('<', end='')

            states_r = states[r, :, :].copy()

            # shuffle the loads that get sampled together
            shuffled_load_IDs = np.random.permutation(num_load)

            for g in range(0, num_load, num_together):

                IDs = shuffled_load_IDs[g:g+num_together]

                loglhoodmtx = np.zeros((num_combos, num_data))
                for unique_combo in np.unique(degenerate_combos, axis=0):
                    """
                    We save computation time by calculating the log likelihood
                    only for unique brightness states and then assigning them
                    to the corresponding states afterwords. For example if we
                    have two states: dark and bright, and we are sampling the
                    joint phototrajectory for fluorophores A and B, then
                    A-bright-B-dark and A-dark-B-bright would have the same
                    log likelihood. Rather than compute this twice we caclulate
                    the log likelihood for one-fluorophore-bright and assign it
                    to both the above joint states.
                    """
                    idx = (degenerate_combos == unique_combo).all(axis=1)
                    for i in range(num_together):
                        states_r[IDs[i], :] = unique_combo[i]
                    brightness = mu_flor @ states_to_pops(states_r, num_states) + mu_back[r]
                    loglhoodmtx[idx, :] = stats.gamma.logpdf(data[r,:], a=brightness, scale=gain)

                # demand final state is photobleached
                loglhoodmtx[:-num_end:, -1] = -np.inf
                loglhoodmtx[-1, -1] = 0

                # softmax the log likelihood matrix to take it out of log space
                lhoodmtx = np.exp(loglhoodmtx - np.max(loglhoodmtx, axis=0))
                lhoodmtx += (loglhoodmtx > -np.inf) * 1e-300  # for numerical stability

                # run forward-filter-backwards-sample algorithm using numba
                trajectory = FFBS(lhoodmtx, pi_comb)

                # convert from combined state space to regular state space
                states_r[IDs, :] = state_combos[trajectory, :].T

            print('>', end='')

            return states_r

        if parpool_status:
            print('+', end='')
            # todo: get rid of warning that pops up
            results = Parallel(n_jobs=-1)(delayed(sample_states_r)(roi) for roi in range(num_rois))
            for roi, states_roi in enumerate(results):
                states[roi, :] = states_roi[:]
        else:
            print('-', end='')
            for roi in range(num_rois):
                states[roi, :] = sample_states_r(roi)

        val.states = states
        val.num_flor = np.sum(states[:, :, 0] < num_states - 1, axis=1)
        val.P = self.posterior(val)

        return

    def sample_transitions(self, val) -> None:
        """
        This function samples a new transition matrix.

        :param val: This parameter is a Simplenamespace object
        containing the stored value of each random variable
        and parameter in the posterior.
        :return: This function returns val, but with an
        updated transition matrix.
        """

        # get values
        states = val.states
        pi_conc = val.transitions_conc
        num_states = val.num_states

        # count the number of each transition that occurs
        counts = np.zeros((num_states + 1, num_states))
        for i in range(num_states):
            counts[-1, i] = np.sum(states[:, :, 0] == i)
            for j in range(num_states):
                counts[i, j] = np.sum((states[:, :, :-1] == i) * (states[:, :, 1:] == j))
        counts[-1, -1] = 0  # fluorophores starting photobleached are interpretted as load off only

        # sample from dirichlet distribution
        val.transitions = Dirichlet.sample(counts + pi_conc)
        val.P = self.posterior(val)

        return

    def plot_variables(self, val=None, roi=None) -> None:
        """
        This funtions plots the data and mean trajectory of for a set
        of variables. This is primarily used for debugging or real
        time feedback for how well the variables are matching the data.

        :param val: This parameter is a Simplenamespace object
        containing the stored value of each random variable
        and parameter in the posterior.
        :param roi: This is the ROI index for plotting.
        :return: This generates a plot but returns nothing.
        """

        if val is None:
            val = self.history.MAP

        if roi is None:
            roi = np.array([0])
        elif np.isscalar:
            roi = np.array([roi])

        data = self.data

        dt = val.dt
        gain = val.gain
        states = val.states
        mu_flor = val.mu_flor
        mu_back = val.mu_back
        num_rois = val.num_rois
        num_load = val.num_load
        num_data = val.num_data
        num_states = val.num_states

        times = np.arange(num_data) * dt

        fig = plt.gcf()
        fig.clf()
        #fig.set_size_inches(12, 6)
        ax = np.empty((len(roi), 1), dtype=object)
        gs = gridspec.GridSpec(nrows=len(roi), ncols=1, figure=fig)
        ax[0, 0] = fig.add_subplot(gs[0, 0])
        for i in range(len(roi)-1):
            ax[i, 0] = fig.add_subplot(gs[i, 0], sharex=ax[0, 0])

        for i, r in enumerate(roi):

            brightness = mu_flor @ states_to_pops(states[r, :, :], num_states) + mu_back[r]

            ax[i, 0].set_title('roi {}: {} flors'.format(r, val.num_flor[r]))
            ax[i, 0].set_xlabel('time (s)')
            ax[i, 0].set_ylabel('brightness (ADU)')
            ax[i, 0].plot(times, data[r, :], color='g', label='data')
            ax[i, 0].plot(times, brightness * gain, color='b', label='sampled')
            ax[i, 0].legend()

        # plt.tight_layout()
        plt.pause(.1)

        return

    def gibbs_sampler(self, data=None, parameters=None, save_name='test', save_path='outfiles/', plot_status=False, log_file=False, **kwargs):
        """
        This function runs a Gibbs sampler algorithm over the posterior to collect
        samples from the posterior.

        :param data: The brightness time traces from the ROIs.
        :param parameters: The specified parameters used for inference.
        :param save_name: A save_name must be specified in order to save
        the posterior samples in hard memory instead of RAM.
        :param save_path: A save_path chooses the location for where to
        save the posterior samples.
        :param plot_status: If debugging, one can specify to plot each
        iteration to visually monitor the convergence to the data.
        :param log_file: A log file can be made to save the status.
        :param kwargs: Any key word specified will override a parameter
        input in parameters
        :return: The function returns nothing, but saves the posterior samples
        in the prespecified file location.
        """
        print('\n{}\n{}\n{}'.format('-'*len(save_name), save_name, '-'*len(save_name)))

        # creates a log file if specified
        if log_file:
            log = save_name + '.log'
            with open(log, 'w') as handle:
                handle.write('[[[[{}]]]]\n'.format(save_name))
                handle.write('starting Gibbs sampler\n')

        # extract values
        if parameters is None:
            parameters = {}
        parameters = {**PARAMETERS, **parameters, **kwargs}

        # data should be a 2d array where each row is the brightness of a different ROI
        data = np.atleast_2d(data)
        self.data = data

        # set variables for gibbs sampler
        np.random.seed(parameters['seed'])  # set RNG
        val = self.initialize_variables(data, parameters)
        num_iter = val.num_iter

        # set history
        self.history = HistoryH5(
            save_name=save_name,
            path=save_path,
            variables=val,
            num_iter=num_iter,
            fields=[
                'num_flor',
                'mu_flor',
                'mu_back',
                'transitions',
                'P',
            ],
        )

        # run the gibbs sampler
        print('starting Gibbs sampler')
        print('parameters:')
        for key in parameters:
            text = str(getattr(val, key)).replace('\n', ', ')
            print('--{} = {}'.format(key, text))
            if log_file:
                with open(log, 'a') as handle:
                    handle.write('--{} = {}\n'.format(key, text))
        for iter_num in range(num_iter):
            print('iteration {} of {} ['.format(iter_num + 1, num_iter), end='')
            t = time.time()
            self.sample_states(val)
            print('%', end='')
            self.sample_mu(val)
            print('%', end='')
            self.sample_transitions(val)
            print('%', end='')
            if plot_status:
                self.plot_variables(val)
                print('%', end='')
            self.history.checkpoint(val, iter_num)
            print('%', end='')
            print('] ({} s)'.format(round(time.time()-t, 2)))
            print('num_flors=[{}]'.format(','.join(str(num_flor) for num_flor in val.num_flor)))
            if log_file:
                with open(log, 'a') as handle:
                    handle.write('iteration {} of {} ({}s)\n'.format(iter_num + 1, num_iter, round(time.time()-t, 2)))
                    handle.write('num_flors=[{}]\n'.format(','.join(str(num_flor) for num_flor in val.num_flor)))

        print('sampling complete')
        if log_file:
            with open(log, 'a') as handle:
                handle.write('sampling complete\n')

        return

