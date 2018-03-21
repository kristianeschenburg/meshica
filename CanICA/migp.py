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

class MIGP(object):

    def __init__(self,object):

        pass

    def __fit(self,input_files):

        pass