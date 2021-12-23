
import numba as nb
import numpy as np
import h5py
import copy
import dill as pickle
from matplotlib import pyplot as plt
from scipy.special import gammaln
from scipy.special import digamma
from scipy.special import gamma as gammafunc
from scipy.special import beta as betafunc

PI = np.pi


@nb.vectorize(nopython=True, cache=True)
def fast_digamma(x):
    """
    The standard python digamma function is slow. Because we use the
    digamma function many times per iteration of the HMC step, it is
    convenient to have a faster function than the built in digamma.
    This faster implementation of the digamma distribution is taken from:
    https://gist.github.com/timvieira/656d9c74ac5f82f596921aa20ecb6cc8
    """
    # Faster digamma function assumes x > 0.
    r = 0
    while x <= 5:
        r -= 1/x
        x += 1
    f = 1 / (x * x)
    t = f*(-1/12.0+f*(1/120.0+f*(-1/252.0+f*(1/240.0+f*(-1/132.0+f*(691/32760.0+f*(-1/12.0+f*3617/8160.0)))))))
    return r + np.log(x) - 0.5/x + t


@nb.njit(cache=True)
def forward_filter_nb(likelihood_matrix, transition_matrix):
    """
    This function calculates the forward filter used for sampling the
    states.

    :param likelihood_matrix: A matrix containing the likelihood value
    at each timepoint for each state
    :param transition_matrix: The transition matrix. The final row is
    the initial starting probability array.
    :return: The forward filter.
    """

    # extract values
    lhood = likelihood_matrix
    pi0 = transition_matrix[-1, :]
    pis = transition_matrix[:-1, :]
    num_states, num_data = lhood.shape

    # find initial probability
    forward = np.zeros((num_states, num_data))
    forward[:, 0] = lhood[:, 0] * pi0
    forward[:, 0] /= np.sum(forward[:, 0])
    for n in range(1, num_data):
        """
        The probability of each state at each time level is the 
        likelihood times the probability of the state marginalized over 
        the state of the previous time step
        """
        forward[:, n] = lhood[:, n] * (pis.T @ forward[:, n - 1])
        forward[:, n] /= np.sum(forward[:, n])

    return forward


@nb.njit(cache=True)
def backward_sample_nb(forward_filter, transition_matrix):
    """
    This funciton takes a forward filter and a transition matrix and
    outputs a sampled state trajectory.

    :param forward_filter: The forward filter.
    :param transition_matrix: The transition matrix where the final
    row is the initial state probability.
    :return: A sampled state trajectory.
    """

    # get values
    forward = forward_filter
    pis = transition_matrix[:-1, :]
    num_states, num_data = forward.shape

    # start by sampling the final state from the forward filter
    states = np.zeros(num_data, dtype=np.int32)
    s = np.searchsorted(np.cumsum(forward[:, -1]), np.random.rand())
    states[-1] = s
    for m in range(1, num_data):
        """
        Every preceeding state is sampled from the forward filter
        times the probability of the preceeding state given
        the following step.
        """
        n = num_data - m - 1
        backward = forward[:, n] * pis[:, s]
        backward /= np.sum(backward)
        s = np.searchsorted(np.cumsum(backward), np.random.rand())
        states[n] = s

    return states


def FFBS(likelihood_matrix, transition_matrix):
    """
    "Forward Filter Backwards Sample"
    This algorithm calculates the forward filter then
    draws a sample from it using the backward sample
    algorithm.

    :param likelihood_matrix: The likelihood of each state
    at each time level.
    :param transition_matrix: The transition matrix where
    the final row is the initial state probability.
    :return: A sampled state trajectory.
    """
    return backward_sample_nb(forward_filter_nb(likelihood_matrix, transition_matrix), transition_matrix)


class HistoryH5:
    """
    This class is used to conveniently save the values of each iteration
    into an hdf5 file.
    """

    def __init__(self, save_name, variables=None, fields=None, num_iter=None, path=''):

        # initialize save name, path, and number of iterations
        self.save_name = save_name
        self.path = path
        self.fields = fields  # note that we do not want to save every feature so we must specify which to save
        self.num_iter = num_iter

        if variables is None:
            # if variables is not specified, then we are loading a previous sampler
            self.load_history()
        else:
            if fields is None:
                fields = variables.__dict__().keys()
                self.fields = fields
            self.initialize(variables)

        return

    def load_history(self):
        """
        This function is used when loading results from a previous sampler.
        """

        save_name = self.save_name
        path = self.path

        # get the saved fields
        h5 = h5py.File(path + save_name, 'r')
        fields = list(h5.keys())
        h5.close()
        self.fields = fields

        # get the number of iterations
        h5 = h5py.File(path + save_name, 'r')
        num_iter = h5['P'].shape[0]
        h5.close()
        self.num_iter = num_iter

        return

    def initialize(self, variables):
        """
        This function creates an h5 file and sets up its attributes.
        Variables is the object containing all relevant variables
        so that we can correcly specify the shape of each attribute
        we wish to save.
        """
        save_name = self.save_name
        path = self.path
        fields = self.fields
        num_iter = self.num_iter

        if num_iter is None:
            raise Exception('num_iter must be specified when saving')

        # the MAP sample is saved in a pickle file
        with open(path + save_name + '_MAP.pickle', 'wb') as handle:
            pickle.dump(variables, handle)

        # create h5 file to save the variables
        h5 = h5py.File(path + save_name, 'w')
        for field in fields:
            # we set the size of a chunk to coincide with the shape of the variable
            variables_field = getattr(variables, field)
            chunk = (1, *np.shape(variables_field))
            if len(chunk) == 1:
                # if the variable is scalar specify a (1,1) chunk size
                chunk = (1, 1)
                dtype = type(variables_field)
            else:
                dtype = variables_field.dtype.type
            shape = (num_iter, *chunk[1:])
            h5.create_dataset(name=field, shape=shape, chunks=chunk, dtype=dtype)
        h5.close()

        return

    def checkpoint(self, variables, iter_num):
        """
        At each iteration, n, of the Gibbs sampler, we
        save the variables in the n'th chunk.
        """

        save_name = self.save_name
        path = self.path
        fields = self.fields

        if iter_num == 0:
            # if this is the first iteration we must initialize the h5 file
            self.initialize(variables)
        else:
            if variables.P >= np.max(self.get('P')[:iter_num]):
                with open(path + save_name + '_MAP.pickle', 'wb') as handle:
                    pickle.dump(variables, handle)

        h5 = h5py.File(path + save_name, 'r+')
        for field in fields:
            h5[field][iter_num, :] = getattr(variables, field)
        h5.close()

        return

    def get(self, field, burn=0, last=None):
        """
        After the sampler is run we can get the results using
        this function. We can also specify a range (burn:last)
        to filter out.
        """

        save_name = self.save_name
        path = self.path

        if self.num_iter is None:
            h5 = h5py.File(path + save_name, 'r')
            self.num_iter = h5['P'].shape[0]
            h5.close()

        if 0 < burn < 1:
            # burn can be an index number, or a number between 0 and 1
            # if burn is between 0 and 1 we take it to mean the fraction of samples to ignore
            burn = round(burn * self.num_iter)

        if field.lower() == 'map':
            # when getting the map we load the map pickle file
            with open(path + save_name + '_MAP.pickle', 'rb') as handle:
                value = pickle.load(handle)
        else:
            h5 = h5py.File(path + save_name, 'r')
            value = h5[field][burn:last, :]
            h5.close()

        return value


class Dirichlet:
    """
    The scipy.stats Dirichlet distribution is inconvenient because
    it does not allow zeros to be input into the array. Under our
    notation, we allow zeros in a vector sampled from a Dirichlet
    distribution with the understanding that we ignore those
    indices. For that reason we create a Dirichlet distriubtion
    class that allows us to sample arrays from concentrations with 0's.
    Furthermore, we allow allow sampling of 2D matrixes with the
    understanding that each row is sampled from a seperate Dirichlet
    distribution.
    """

    def __init__(self):
        pass

    @staticmethod
    def sample(concentration):
        X = np.random.gamma(shape=concentration)
        for k in range(X.shape[0]):
            X[k, :] /= np.sum(X[k, :])
        return X

    @staticmethod
    def logpdf(X, conc):
        J, K = X.shape
        prob = 0
        for j in range(J):
            for k in range(K):
                prob += gammaln(np.sum(conc[j, :]))
                if (X[j, k] > 0) and (conc[j, k]):
                    prob += (conc[k] - 1) * np.log(X[j, k]) - gammaln(conc[j, k])

        return prob