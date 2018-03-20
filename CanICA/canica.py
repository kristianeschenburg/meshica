import niio
from operator import itemgetter

import numpy as np
from nilearn.signal import clean
from nilearn.decomposition.base import fast_svd

from scipy.stats import scoreatpercentile
from sklearn.decomposition import fastica
from sklearn.externals.joblib import Memory, delayed, Parallel
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd

class CanICA(object):

    def __init__(self,n_components=20,pca_filter=False,n_init=10,
                 do_cca=False,standardize=True,low_pass=None,high_pass=None,tr=None,
                 threshold=None,random_state=None):

        """

        :param n_components: number of ICA components to generate
        :param pca_filter: apply temporal dimensionality reduction
                            prior to group ICA
        :param n_init: number of times FastICA is restarted
        :param do_cca: whether ors not to run Canonical Correlation Analysis after PCA
        :param standardize: boolean, apply signal cleaning
        :param low_pass: low pass filter
        :param high_pass: high pass filter
        :param tr: repetitiion time
        """

        self.n_components = n_components
        self.pca_filter = pca_filter
        self.n_init = n_init
        self.do_cca = do_cca

        self.standardize = standardize
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.tr = tr

        self.threshold=threshold
        self.random_state=random_state

    def fit(self,input_files):

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

    def _merge_and_reduce(self,input_files):

        """

        Clean, temporally reduce, and concatenate resting state matrix files.

        :param input_files: list of input resting state matrix files
        :return signals: concatenated resting state arrays
        """

        signals = []

        for inp in input_files:

            print 'Loading {:}'.format(inp.split('/')[-1])

            matrix = niio.load(inp)
            matrix = clean(matrix,standardize=self.standardize,
                           low_pass=self.low_pass,high_pass=self.high_pass,
                           t_r=self.tr)

            if self.pca_filter:
                matrix = self._reduce(matrix)

            signals.append(matrix)

        signals = np.column_stack(signals)

        return signals.T


    def _raw_fit(self, data):

        """
        Base method for performing group ICA.

        :param data: raw resting state signals
        """

        print 'Fitting with CCA == {:}'.format(str(self.do_cca))

        if self.do_cca:
            S = np.sqrt(np.sum(data ** 2, axis=1))
            S[S == 0] = 1
            data /= S[:, np.newaxis]

        self.components_, self.variance_, _ = randomized_svd(data.T, n_components=self.n_components,
            transpose=True, random_state=None, n_iter=3)

        if self.do_cca:
            data *= S[:, np.newaxis]

        self.components_ = self.components_.T

    def _reduce(self,signals):

        """

        Perform temporal dimensionality reduction.

        :param signals: single-subject resting state matrix.
        :return U: reduced resting state matrix
        """

        U, S, V = fast_svd(signals.T,self.n_components)
        U = U.T.copy()
        U = U * S[:, np.newaxis]
        return U