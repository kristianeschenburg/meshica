from loaded import loaded
from operator import itemgetter

import numpy as np
from nilearn.signal import clean
from nilearn.decomposition.base import fast_svd

from scipy.stats import scoreatpercentile
from scipy.linalg import eigh
from sklearn.decomposition import fastica
from sklearn.externals.joblib import Memory, delayed, Parallel
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd

import random

class MIGP(object):

    def __init__(self,n_components=20,m_eigen=9600,s_init=3,n_init=10,
                 standardize=True,low_pass=None,high_pass=None,t_r=None,
                 threshold=None,random_state=None,):

        """

        Class to run iterative group ICA on 2D surface mesh data using MIGP algorithm
        described in Smith et al. 2014.

         * https://www.ncbi.nlm.nih.gov/pubmed/25094018

        :param n_components: number of ICA components to generate
        :param m_eigen: number of spatial PCA eigenvectors to update
        :param s_init: number of datasets to initialize MIGP with
        :param n_init: number of times FastICA is restarted
        :param standardize: boolean to normalize data
        :param low_pass: low-pass filter
        :param high_pass: high-pass filter
        :param t_r: repetitiion time
        :param threshold:
        :param random_state: random number generator
        """

        self.n_components = n_components
        self.m_eigen = m_eigen
        self.s_init = s_init
        self.n_init = n_init

        self.standardize = standardize
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r

        self.threshold=threshold
        self.random_state=random_state

    def fit(self,input_files):

        """

        :param input_files:
        :return:
        """

        random.shuffle(input_files)
        self._raw_fit(input_files)
        self._unmix_components()
        return self

    def _unmix_components(self):

        """
        Core function of CanICA to rotate components to maximize independance
        """

        print 'Unmixing components'

        random_state = check_random_state(self.random_state)

        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        results = Parallel(n_jobs=4)(
            delayed(fastica)(self.components_.T,whiten=True,
                             fun='cube',random_state=seed) for seed in seeds)

        ica_maps_gen_ = (result[2].T for result in results)
        ica_maps_and_sparsities = ((ica_map,
                                    np.sum(np.abs(ica_map), axis=1).max())
                                   for ica_map in ica_maps_gen_)
        ica_maps, _ = min(ica_maps_and_sparsities, key=itemgetter(-1))

        # Thresholding
        ratio = None
        if isinstance(self.threshold, float):
            ratio = self.threshold
        elif self.threshold == 'auto':
            ratio = 1.
        elif self.threshold is not None:
            raise ValueError("Threshold must be None, "
                             "'auto' or float. You provided %s." %
                             str(self.threshold))

        if ratio is not None:
            abs_ica_maps = abs(ica_maps)
            threshold = scoreatpercentile(
                abs_ica_maps,
                100. - (100. / len(ica_maps)) * ratio)
            ica_maps[abs_ica_maps < threshold] = 0.
        self.components_ = ica_maps

        # flip signs in each component so that peak is +ve
        for component in self.components_:
            if component.max() < -component.min():
                component *= -1

        self.components_ = self.components_.T

    def _raw_fit(self,input_files):

        """

        :param input_files:
        :return:
        """

        W = []
        for s in np.arange(self.s_init):
            W.append(self._merge_and_reduce(input_files[s]))

        # Compute initial estimate of spatial eigenvectors
        print 'Computing initial estimate for {:} subjects.'.format(self.s_init)
        W = np.row_stack(W)
        W = self._estimate(W)

        for s in np.arange(self.s_init,len(input_files)):

            update_data = self._merge_and_reduce(input_files[s])
            W = np.row_stack([W,update_data])

            temporal, variance, spatial = randomized_svd(W, self.m_eigen, n_iter=3)
            W = np.dot(np.diag(variance),spatial)

        self.components_ = W[0:self.n_components,:]

    def _merge_and_reduce(self,input_file):


        print 'Loading {:}'.format(input_file.split('/')[-1])

        matrix = loaded.load(input_file)
        matrix = clean(matrix, standardize=self.standardize,
                       low_pass=self.low_pass, high_pass=self.high_pass,
                       t_r=self.t_r)

        return matrix.T

    def _estimate(self,signals):

        """

        Compute weighted spatial eigenvectors of signal.

        :param signals:
        :return:
        """

        _,variance,spatial = randomized_svd(signals, self.m_eigen, n_iter=3)

        return np.dot(np.diag(variance),spatial)