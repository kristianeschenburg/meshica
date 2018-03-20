import niio
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class DualRegression(object):

    def __init__(self,temporal_standardize=True,z_threshold=2):

        """

        :param temporal_standardize: boolean indicating whether to variance-
                                    noramalize design matrix for
                                    temporal regression
        :param z_threshold: value at which to high-pass threshold spatial components
                            If None, returns raw components.
        :return:
        """

        self.z_threshold = z_threshold
        self.temporal_standardize = temporal_standardize

    def fit(self,input_files,group_components):

        """

        :param input_files:
        :param group_components:
        :return:
        """

        try:
            assert group_components.shape[1] > 0
        except:
            raise ValueError('Must fit group components first.')

        merged = self._merge_and_reduce(input_files)

        self.time_series = self._temporal(merged,group_components)
        self.spatial_components = self._spatial(merged,self.time_series)

    def _merge_and_reduce(self,input_files):

        """

        Load resting state matrix files.

        :param input_files: list of input resting state matrix files
        :return signals: concatenated resting state arrays
        """

        signals = []

        for inp in input_files:

            print 'Loading {:}'.format(inp.split('/')[-1])

            matrix = niio.load(inp)
            signals.append(matrix)

        return signals

    def _temporal(self,signals,group_components):

        """

        :param signals:
        :param group_components:
        :return:
        """

        print 'Temporal regression.'

        models = {}.fromkeys(np.arange(len(signals)))
        time_series = []

        for j,signal in enumerate(signals):

            models[j] = LinearRegression()
            models[j].fit(group_components,signal)
            time_series.append(models[j].coef_)

        return time_series

    def _spatial(self,signals,time_components):

        """

        :param signals:
        :param time_components:
        :return:
        """

        print 'Spatial regression.'

        models = {}.fromkeys(np.arange(len(signals)))
        S = StandardScaler(with_mean=False, with_std=True)
        Z = StandardScaler(with_mean=True,with_std=True)
        spatial = []

        for j,signal in enumerate(signals):

            if self.temporal_standardize:
                transformed = S.fit_transform(time_components[j])
            else:
                transformed = time_components[j]

            models[j] = LinearRegression()
            models[j].fit(transformed,signal.T)

            coefficients = models[j].coef_

            if self.z_threshold:
                coefficints = Z.fit_transform(coefficients)
                idx = np.abs(coefficients) < self.z_threshold
                coefficients[idx] = 0

            spatial.append(coefficients)

        return spatial