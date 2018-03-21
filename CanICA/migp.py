import niio
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

    def __init__(self,n_components=20,m_eigen=5500,s_init=3,n_init=10,
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

    def _fit(self,input_files):

        """

        :param input_files:
        :return:
        """

        random.shuffle(input_files)

    def _raw_fit(self):

        pass

    def _merge_and_reduce(self,input_files):

        pass

    def _update(self):