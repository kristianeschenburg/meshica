from niio import loaded, write
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from nilearn.signal import clean

from statsni.confidence import hpd_grid as hpd

class Regressor(object):

    def __init__(self, standardize=True, hdr_alpha=0.05, tr=0.720, low_pass=None, high_pass=None, s_filter=False):

        """
        Class to perform dual regression of group-ICA components.

        Parameters:
        - - - - -
        standardize: bool
            apply temporal normalization
        hdr_alpha: float
            Bayesian confidence interval alpha value
            default: 0.05
        tr: float
            repetition time
        low / high pass: float
            low and high frequency thresholds for spectral filtering
        """

        self.standardize = standardize
        self.s_filter = s_filter
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.tr = tr
        self.hdr_alpha = hdr_alpha

    def fit(self, input_rest, gica_components):

        """
        Wrapper method to run dual regression.

        Parameters:
        - - - - -
        input_rest: float, array
            numpy array of resting-state time-series for single subject
        gica_components: float, array
            numpy array of group-ICA components
        """

        try:
            assert gica_components.shape[1] > 0
        except:
            raise ValueError('Must fit group components first.')

        input_rest = self._merge_and_reduce(input_rest)

        temporal_components = self.temporal_regression(input_rest, gica_components)
        spatial_components = self.spatial_regression(input_rest, temporal_components)

        self.temporal_ = temporal_components
        self.spatial_ = spatial_components

    def _merge_and_reduce(self, signals):

        """

        Apply filtering (optional) to resting state signal.

        Parameters:
        - - - - -
        time_series: float, array
            raw resting-state matrix
        """

        if self.s_filter:

            signals = clean(signals,
                            standardize=self.standardize,
                            low_pass=self.low_pass,
                            high_pass=self.high_pass,
                            t_r=self.tr)
        
        return signals

    def temporal_regression(self, signal, group_components):

        """
        Perform temporal regression.

        :param signals:
        :param group_components:
        :return:
        """

        model = LinearRegression()
        model.fit(group_components, signal)
        
        temporal_coefficients = model.coef_

        return temporal_coefficients

    def spatial_regression(self, signals, temporal_coefficients):

        """
        Perform spatial regression.

        Parameters:
        - - - - -
        signals: float, array
            numpy array of resting-state signals
        temporal_coefficients: float, array
            estimated temporal coefficients of each components
        """

        S = StandardScaler(with_mean=False, with_std=True)
        Z = StandardScaler(with_mean=True, with_std=True)

        if self.standardize:
            temporal_coefficients = S.fit_transform(temporal_coefficients)

        model = LinearRegression()
        model.fit(temporal_coefficients, signals.T)

        spatial_coefficients = model.coef_
        spatial_coefficients = Z.fit_transform(spatial_coefficients)

        if self.hdr_alpha:
            [bounds, _, _, _] = hpd(spatial_coefficients, alpha=self.hdr_alpha)
            lower = bounds[0][0]
            upper = bounds[0][1]
            idx = np.asarray(np.logical_or(spatial_coefficients <= lower, spatial_coefficients >= upper))
            spatial_coefficients = (idx*spatial_coefficients)

        return spatial_coefficients
