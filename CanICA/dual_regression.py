import niio
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class DualRegression(object):

    def __init(self,temporal_standardize=True,spatial_standardize=False):

        """

        :param temporal_standardize: boolean indicating whether to variance
                                    noramalize design matrix for
                                    temporal regression

        :return:
        """

        self.temporal_standardize = temporal_standardize
        self.spatial_standardize = spatial_standardize

    def fit(self,input_files,group_components):

        """

        :param input_files:
        :param group_components:
        :return:
        """

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

        models = {}.fromkeys(np.arange(len(signals)))
        S = StandardScaler(with_mean=False, with_std=True)
        spatial = []

        for j,signal in signals:

            if self.temporal_standardize:
                transformed = S.fit_transform(time_components[j])
            else:
                transformed = time_components[j]

            models[j] = LinearRegression()
            models[j].fit(transformed,signal.T)

            spatial.append(models[j].coef_)

        return spatial