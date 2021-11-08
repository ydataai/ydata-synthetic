"""
Scaler with mode specific normalization.
"""
from typeguard import typechecked
from pandas import DataFrame
from numpy import uint8, sqrt, zeros, apply_along_axis, where, array, concatenate
from numpy.random import choice
from scipy.stats import norm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import BayesianGaussianMixture

class ModeSpecificNormalizer(TransformerMixin, BaseEstimator):

    @typechecked
    def __init__(self, max_modes: int = 10, weight_threshold: float = 5e-3):
        self._max_modes = max_modes
        self._weight_threshold = weight_threshold
        self._modes = {}

    @typechecked
    def fit(self, X: DataFrame):
        """Fits the GaussianMixture to the passed data.
        Stores parameters of all found modes, used to scale and unscale the data.

        Arguments:
            X (pd.DataFrame): Data used to fit the mixture.
        """
        for feature in list(X):
            vgm = BayesianGaussianMixture(
                n_components = self._max_modes,
                weight_concentration_prior_type = 'dirichlet_process',
                weight_concentration_prior = 0.001,
                n_init = 3,
                max_iter = 250,
                tol = 5e-2
            )
            vgm.fit(X[feature].values.reshape(-1, 1))  # Is reshape needed here?

            relevant_modes = vgm.weights_ > self._weight_threshold

            self._modes[feature] = {'means': vgm.means_[relevant_modes].reshape(-1),
                                    'stdev': sqrt(vgm.covariances_[relevant_modes].reshape(-1)),}

    @typechecked
    def transform(self, X: DataFrame):
        """Transforms a batch of data with same schema as the fitted data.
        The returned rows are the same as X. Columns are n_features blocks of n_modes + 1 columns.

        Arguments:
            X (pd.DataFrame): A batch of data to be transformed.

        Returns:
            transformed (np.array): The batch converted feature-wise to one-hot mode masks and normalized values."""
        assert set(X.columns) == set(self._modes.keys()), "The provided DataFrame is not compatible with this transformer schema."
        transformed = []
        for feature in list(X):
            modes = self._modes[feature]
            n_modes = len(modes['means'])
            p_mat = zeros((X.shape[0], n_modes))
            ohe = zeros((X.shape[0], n_modes), dtype=uint8)
            for i in range(n_modes):
                mean = modes['means'][i]
                stdev = modes['stdev'][i]
                p_mat[:, i] = X[feature].apply(lambda x: norm.pdf(x, loc = mean, scale = stdev)).values
            p_mat = p_mat/p_mat.sum(1)[:,None]
            ohe = apply_along_axis(self._sample_p, 1, p_mat, n_modes = n_modes)
            vals = self._mode_norm(X[feature].values, ohe, feature)
            transformed += [ohe, vals]
        return concatenate(transformed, axis=1)

    def _sample_p(self, p_vector, n_modes):
        ohe_ = zeros(n_modes, dtype=uint8)
        ohe_[choice(n_modes, p = p_vector)] = 1
        return ohe_

    def _mode_norm(self, vals, ohe_array, feature, mode = 'norm'):
        modes_i = where(ohe_array==1)[1]
        norm_params = zeros((vals.shape[0], 2))
        norm_params = apply_along_axis(func1d=self._params_lookup, arr=modes_i, feature=feature, axis=0).T
        if mode == 'norm':
            return ((vals - norm_params[:, 0]) / (4*norm_params[:, 1])).reshape(-1, 1)
        elif mode == 'inverse':
            return ((vals * 4 * norm_params[:, 1]) + norm_params[:, 0]).reshape(-1, 1)

    def  _params_lookup(self, i, feature):
        return array([self._modes[feature]['means'][i], self._modes[feature]['stdev'][i]])

    def inverse_transform(self, X: array):
        """Inverts the mode specific normalization.
        The one hot vector is used to define the mode normalization to invert value.

        Arguments:
            X (np.array): A batch of data transformed by mode specific scaling.

        Returns:
            Xt (np.array): The batch of data in its original scale.
        """
        unscaled = []
        for feat, parameters in self._modes.items():
            n_modes = len(parameters['means'])
            mode, X = X[:, :n_modes + 1], X[:, n_modes + 1:]
            ohe, vals = mode[:, :-1], mode[:, -1]
            vals = self._mode_norm(vals, ohe, feat, 'inverse')
            unscaled.append(vals)
        return concatenate(unscaled, axis=1)
