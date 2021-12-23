
import sys
import numpy as np
import matplotlib.pyplot as plt
from fluorophore_counter import FluorophoreCounter

path_data = 'data/'

def main(ID=0, seed=0):

    print('getting data')
    data_files = [
        'data_140_binding_sites',
        'data_80_binding_sites',
        'data_70_binding_sites',
        'data_40_binding_sites',
        'data_35_binding_sites',
        'data_20_binding_sites',
        'sim_100_flors',
        'sim_80_flors',
        'sim_60_flors',
        'sim_40_flors',
        'sim_20_flors',
        'sim_gain2200',
        'sim_start_dark',
        'sim_D0B1',
        'sim_D1B1',
        'sim_D1B2',  # base
    ]
    data_file = data_files[ID]

    # get data
    data = np.genfromtxt('{}{}.csv'.format(path_data, data_file), delimiter=',')

    # set parameters
    num_states = 4
    num_iter = 20000
    parameters = {
        'seed': seed,
        'dt': .05,
        'gain': 22,
        'num_states': num_states,
        'flor_brightness_guess': 10000,
        'parpool_status': True,
    }
    save_path = 'outfiles/'
    save_name = 'learned_{}'.format(data_file)

    # run the fluorophore counter
    print('starting sampler')
    counter = FluorophoreCounter()
    counter.gibbs_sampler(
        data=data,
        num_iter=num_iter,
        parameters=parameters,
        save_name=save_name,
        save_path=save_path,
        plot_status=False,
        log_file=False,
    )

    print('...')


if __name__ == "__main__":

    print('starting...')

    ID = 0
    seed = 0
    if len(sys.argv) > 1:
        ID = int(sys.argv[1])
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])

    print('running main')
    main(ID=ID, seed=seed)

    print('done')


