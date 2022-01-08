# FluorophoreCounter

The flourophore counter algorithm has been packaged into a python class. To import the fluorophore counter use:

from fluorophore_counter import FluorophoreCounter

The algorithm as some required parameters. Before running the fluorophore counter you must create a dictionary and specify the camera gain and fluorophore brightness guess in ADU (which can be estimated by eye from the last photobleaching brightness drop). Parameters can also take in other options such as "seed" which sets the random number generator seed, "parpool_status" which is a bool determining whether or not to use paralelization, "dt" which is the exposure period of the camera (which affects plotting but not the inferreed number of fluorophores), and "num_states" which specifies the state model that will be used. Other hyperparameters found in the dictionary on top of fluorophore_counter.py can be tuned, but that is not reccomended. The reccomended parameters setting is

parameters = {
    'dt': <<Insert camera exposure period here>>
    'gain': <<Insert calibrated gain here>>,
    'flor_brightness_guess': <<Insert brightness guess here>>,
    'num_states': 4,         # num_states should always be greater than 2 to include a blinking state
    'parpool_status': True,  # Use True only if you wish to paralelize.
}

To run the algorithm on a data set we use the class method, gibbs_sampler. gibbs_sampler takes in two required arguments, data and parameters, as well as other useful arguments. In particular, num_iter sets the number of iterations of MCMC will be used, save_name sets the name of the saved MCMC runs, and plot_status allows the user to choose whether or not to see live plots of the MCMC chain. The recomended way to run the algorithm is

counter = FluorophoreCounter()
counter.gibbs_sampler(
    data=data,
    parameters=parameters,
    num_iter=10000,
    save_name='my_data',  # change the name as you look at new data sets or use different parameters
    plot_status=True,  # a plot of the data will be generated at each iteration if plot_status is True
)

There is no output from gibbs_sampler, instead a set of files is saved in the local directory with names determined by save_name (to change the output directory one can specify save_path in gibbs_sampler). The MCMC samples can be accessed from a class attribute called history. history is a class that organizes MCMC samples into H5 and pickle files. To get the sampled number of fluorophores from the MCMC chain, history has a built in function called "get" which takes in a string of the desired variable and then outputs an array of the samples. The first dimension of the output from "get" will always the the number of iterations. For most purposes we only need to find the MCMC chain of the number of fluorophores in each ROI. To get this estimate we run

mcmc_chain_for_num_flors = counter.history.get('num_flor')

mcmc_chain_for_num_flors will be a <number of iterations> X <number of ROIs> array. We can also access the mcmc chains for the fluorophore state brightnesses, the background brightnesses, the transition matrix, and the log probability using "get". Lastly, we can use "get" to get the MAP set of variables, that is, the set of variables encountered in the MCMC chain that had the highest probability. To see this we use

map_vars = counter.history.get('map')

map_vars will be a SimpleNamespace type where each attribute is a parameter used in the Gibbs sampler. For example, to get the highest probability number of fluorophores in each ROI we can use

map_num_flors = map_vars.num_flor

Puttining it all together, assuming that all we want from the algorithm is the number of fluorophore in the data, we can use 



from fluorophore_counter import FluorophoreCounter

# set parameters (modify these three lines as needed)
data = np.genfromtxt('data.txt')
gain = 50
brightness_guess = 10000
parameters = {
    'gain': gain,
    'flor_brightness_guess': brightness_guess,
    'num_states': 3,
}

# run gibbs sampler
counter = FluorophoreCounter()
counter.gibbs_sampler(
    data=data,
    parameters=parameters,
    num_iter=10000,
    save_name='my_data',
    plot_status=True,
)

# get output
map_num_flors = counter.history.get('map').num_flors


