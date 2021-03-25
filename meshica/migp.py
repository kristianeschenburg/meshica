from niio import loaded, write
from operator import itemgetter

import numpy as np
from nilearn.signal import clean
from nilearn.decomposition.base import fast_svd

from scipy.stats import scoreatpercentile
from scipy.linalg import eigh
from sklearn.decomposition import fastica

from joblib import Parallel, delayed
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd

import random

class MIGP(object):

    def __init__(self, n_components=10, m_eigen=9600, s_init=3, n_init=10,
                 standardize=True, low_pass=None, high_pass=None, t_r=None,
                 threshold=None, random_state=None, mask=None):

        """

        Class to run iterative group ICA on 2D surface mesh data using MIGP algorithm
        described in Smith et al. 2014.

         * https://www.ncbi.nlm.nih.gov/pubmed/25094018

         Parameters:
         - - - - - -
        n_components: int
            number of ICA components to generate
        m_eigen: int
            number of spatial PCA eigenvectors to update
        s_init: int
            number of datasets to initialize MIGP with
        n_init: int
            number of times FastICA is restarted
        standardize: bool
            boolean to normalize data
        low_pass: low-pass filter
        high_pass: high-pass filter
        t_r: float
            repetitiion time (TR)
        threshold: float, >=0, <=1
            threshold value of coefficient maps
        random_state: int
            random number generator
        mask: int array
            boolean mask, indicating which voxel to keep
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

        self.mask = mask

    def fit(self, input_files):

        """

        :param input_files:
        :return:
        
        """

        random.shuffle(input_files)
        self._raw_fit(input_files)
        self._unmix_components()

    def _unmix_components(self):

        """
        Core function of CanICA to rotate components to maximize independance
        """

        print('Unmixing components')

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

        # flip signs in each component so that peak is (+)
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
        for temp_file in input_files[:self.s_init]:

            temp_matrix = loaded.load(temp_file)
            nans = np.isnan(temp_matrix).sum()
            infs = np.isinf(temp_matrix).sum()

            if nans > 0 or infs > 0:
                print('%s has %i NANs and %i INFs' % (temp_file, nans, infs))
                continue
            else:
                W.append(self._merge_and_reduce(temp_matrix))

        # Compute initial estimate of spatial eigenvectors
        print('Computing initial estimate for {:} subjects.'.format(self.s_init))
        W = np.row_stack(W).squeeze()
        print('Initialization matrix: {:}'.format(W.shape))
        W = self._estimate(W)

        print('Initial estimate shape: {:}'.format(W.shape))

        for k, temp_file in enumerate(input_files[self.s_init:]):

            print('Adding file # %i' % (k+1+self.s_init))

            temp_matrix = loaded.load(temp_file)
            nans = np.isnan(temp_matrix).sum()
            infs = np.isinf(temp_matrix).sum()

            if nans > 0 or infs > 0:
                print('%s has %i NANs and %i INFs' % (temp_file, nans, infs))
                continue
            else:
                print('Adding file: %s' % (temp_file))
                update_data = self._merge_and_reduce(temp_matrix)
                W = np.row_stack([W, update_data])

                temporal, variance, spatial = randomized_svd(W, self.m_eigen, n_iter=3)
                W = np.dot(np.diag(variance), spatial)

        self.components_ = W[0:self.n_components, :]

    def _merge_and_reduce(self, matrix):

        if self.mask is not None:
            if self.mask.shape[0] != matrix.shape[0]:
                raise('Mask must have the same number of samples as the matrix.')
            else:
                matrix = matrix[np.where(self.mask),:]

        print(matrix.shape)

        matrix = clean(matrix, standardize=self.standardize,
                       low_pass=self.low_pass, high_pass=self.high_pass,
                       t_r=self.t_r)

        return matrix.T.squeeze()

    def _estimate(self, signals):

        """

        Compute weighted spatial eigenvectors of signal.

        :param signals:
        :return:
        """

        _,variance,spatial = randomized_svd(signals, self.m_eigen, n_iter=3)

        return variance[:, None]*spatial