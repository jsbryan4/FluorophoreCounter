
import numpy as np
import matplotlib.pyplot as plt
from tools.bayes_tools import BayesTools
from fluorophore_counter import FluorophoreCounter

xx = BayesTools()
save_path = 'data/'

# create base parameters
base_parameters = {
    # simulation parameters
    'seed': 1993,       # seed for random number generator
    'num_sites':  20,   # number of binding sites
    'binding_eff': .7,  # binding efficiency
    # constants
    'dt': .05,    # time step
    'gain': 22,   # gain
    # variables
    'mu_back': 100,
    'mu_flor': np.array([0, 450, 350, 0]),
    'transitions': np.array(
        [
            [0.9950, 0.0025, 0.0025, 0.0000],
            [0.0005, 0.9810, 0.0180, 0.0005],
            [0.0005, 0.0190, 0.9800, 0.0005],
            [0.0000, 0.0000, 0.0000, 1.0000],
            [0.0000, 0.5000, 0.5000, 0.0000],
        ]
    ),
    # numbers
    'num_rois': 50,      # number of ROIs
    'num_data': 20000,   # number of time levels
    'num_states': 4,     # number of states
}


def simulate_data(save_name_, parameters_):
    parameters_ = {
        **base_parameters,
        **parameters_,
    }
    data, ground_truth = FluorophoreCounter.simulate_data(parameters_)
    np.savetxt(save_path + save_name_ + '.csv', data, delimiter=',')
    xx.save_pickle(ground_truth, save_path + save_name_)
    return


# sim base case
simulate_data(
    'sim_base',
    {},
)

# sim start dark
simulate_data(
    'sim_start_dark',
    {
        'transitions': np.array(
            [
                [0.9950, 0.0025, 0.0025, 0.0000],
                [0.0005, 0.9810, 0.0180, 0.0005],
                [0.0005, 0.0190, 0.9800, 0.0005],
                [0.0000, 0.0000, 0.0000, 1.0000],
                [0.4000, 0.3000, 0.3000, 0.0000],
            ]
        ),
    },
)

# sim high noise
gain = 2200
amp = gain / base_parameters['gain']
simulate_data(
    'sim_gain{}'.format(gain),
    {
        'gain': 2200,
        'mu_back': base_parameters['mu_back'] / amp,
        'mu_flor': base_parameters['mu_flor'] / amp,
    },
)

# sim num_flors
for num_flors in [20, 40, 60, 80, 100]:
    simulate_data(
        'sim_{}_flors'.format(num_flors),
        {
            'num_sites': 2 * num_flors,
            'binding_eff': .5,
        }
    )

# sim 0 dark
simulate_data(
    'sim_D0B1',
    {
        'num_states': 2,
        'mu_flor': np.array([450, 0]),
        'transitions': np.array(
            [
                [0.9990, 0.0005],
                [0.0000, 1.0000],
                [1.0000, 0.0000],
            ]
        ),
    }
)

# sim 1 brights
simulate_data(
    'sim_D1B1',
    {
        'num_states': 3,
        'mu_flor': np.array([0, 450, 0]),
        'transitions': np.array(
            [
                [0.9950, 0.0050, 0.0000],
                [0.0005, 0.9990, 0.0005],
                [0.0000, 0.0000, 1.0000],
                [0.0000, 1.0000, 0.0000],
            ]
        ),
    }
)

# sim 2 brights
simulate_data(
    'sim_D1B2',
    {
        'num_states': 4,
        'mu_flor': np.array([0, 450, 350, 0]),
        'transitions': np.array(
            [
                [0.9950, 0.0025, 0.0025, 0.0000],
                [0.0005, 0.9810, 0.0180, 0.0005],
                [0.0005, 0.0190, 0.9800, 0.0005],
                [0.0000, 0.0000, 0.0000, 1.0000],
                [0.0000, 0.5000, 0.5000, 0.0000],
            ]
        ),
    }
)



