from niio import loaded, write
from operator import itemgetter

import numpy as np
from nilearn.signal import clean
from nilearn.decomposition.base import fast_svd

from scipy.stats import scoreatpercentile
from sklearn.decomposition import fastica
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd

from statsni.confidence import hpd_grid as hpd

import joblib
from joblib import Memory, delayed, Parallel

class ICA(object):
    
    def __init__(self, n_components=20, pca_filter=False, n_init=10,
                 do_cca=False,standardize=True, low_pass=None, high_pass=None, t_r=None,
                 threshold='auto', random_state=None, hdr_alpha=0.05):

        """

        Class to run single-subject ICA on 2D surface mesh data.  
        Based on original nilearn implementation
        for 4D data.

         * https://www.ncbi.nlm.nih.gov/pubmed/20153834

        :param n_components: number of ICA components to generate
        :param pca_filter: apply temporal dimensionality reduction
                            prior to group ICA
        :param n_init: number of times FastICA is restarted
        :param do_cca: boolean to run Canonical Correlation Analysis after PCA
        :param standardize: boolean to normalize data
        :param low_pass: low-pass filter limit
        :param high_pass: high-pass filter limit
        :param tr: repetitiion time
        """

        self.n_components = n_components
        self.pca_filter = pca_filter
        self.n_init = n_init
        self.do_cca = do_cca

        self.standardize = standardize
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r

        self.threshold=threshold
        self.random_state=random_state

    def fit(self, input_files):

        """
        Wrapper method for performing group ICA.

        :param input_files: list of input resting state matrices
        """

        signals = self._merge_and_reduce(input_files)
        self._raw_fit(signals)
        self._unmix_components()
        return self

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

        # flip signs in each component so that peak is +ve
        for component in self.components_:
            if component.max() < -component.min():
                component *= -1

        self.components_ = self.components_.T


    def _merge_and_reduce(self, input_file):

        """
        Clean, temporally reduce, and concatenate resting state matrix files.

        :param input_files: list of input resting state matrix files
        :return signals: concatenated resting state arrays
        """

        print('Loading {:}'.format(input_file.split('/')[-1]))

        matrix = loaded.load(input_file)
        if matrix.shape[0] < matrix.shape[1]:
            matrix = matrix.T
        matrix = clean(matrix,standardize=self.standardize,
                        low_pass=self.low_pass, high_pass=self.high_pass,
                        t_r=self.t_r)

        if self.pca_filter:
            matrix = self._reduce(matrix)

        return matrix.T


    def _raw_fit(self, data):

        """
        Base method for performing group ICA.

        :param data: raw resting state signals
        """

        print('Fitting with CCA = {:}'.format(str(self.do_cca)))

        if self.do_cca:
            S = np.sqrt(np.sum(data ** 2, axis=1))
            S[S == 0] = 1
            data /= S[:, np.newaxis]

        self.components_, self.variance_, _ = randomized_svd(data.T, n_components=self.n_components,
            transpose=True, random_state=None, n_iter=3)

        if self.do_cca:
            data *= S[:, np.newaxis]

        self.components_ = self.components_.T

    def _reduce(self, signals):

        """
        Perform temporal dimensionality reduction.

        :param signals: single-subject resting state matrix.
        :return U: reduced resting state matrix
        """

        U, S, V = fast_svd(signals.T, self.n_components)
        U = U.T.copy()
        U = U * S[:, np.newaxis]
        return U