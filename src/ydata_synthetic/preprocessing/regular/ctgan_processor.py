from __future__ import annotations

from typing import List, Optional
from typeguard import typechecked
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError, ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import OneHotEncoder

from ydata_synthetic.preprocessing.base_processor import BaseProcessor

@dataclass
class ColumnMetadata:
    """
    Dataclass that stores the metadata of each column.
    """
    start_idx: int
    end_idx: int
    discrete: bool
    output_dim: int
    model: any
    components: list
    name: str


@typechecked
class CTGANDataProcessor(BaseProcessor):
    """
    CTGAN data preprocessing class.
    It works like any other transformer in scikit-learn with the methods fit, transform and inverse_transform.
    Args:
        n_clusters (int), default=10:
            Number of clusters.
        epsilon (float), default=0.005:
            Epsilon value.
        num_cols (list of strings):
            List of names of numerical columns.
        cat_cols (list of strings):
            List of names of categorical columns.
    """
    SUPPORTED_MODEL = 'CTGAN'

    def __init__(self, n_clusters=10, epsilon=0.005, 
                 num_cols: Optional[List[str]] = None, 
                 cat_cols: Optional[List[str]] = None):
        super().__init__(num_cols, cat_cols)

        self._n_clusters = n_clusters
        self._epsilon = epsilon
        self._metadata = None
        self._dtypes = None
        self._output_dimensions = None
    
    @property
    def metadata(self) -> list[ColumnMetadata]:
        """
        Returns the metadata for each column.
        """
        return self._metadata
    
    @property
    def output_dimensions(self) -> int:
        """
        Returns the dataset dimensionality after the preprocessing.
        """
        return int(self._output_dimensions)

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X: pd.DataFrame) -> CTGANDataProcessor:
        """
        Fits the data processor to a passed DataFrame.

        Args:
            X (DataFrame):
                DataFrame used to fit the processor parameters.
                Should be aligned with the num/cat columns defined in initialization.
        Returns:
            self (CTGANDataProcessor): The fitted data processor.
        """
        self._dtypes = X.infer_objects().dtypes
        self._metadata = []
        cur_idx = 0
        for column in X.columns:
            column_data = X[[column]].values
            if column in self.cat_cols:
                ohe = OneHotEncoder(sparse_output=False)
                ohe.fit(column_data)
                n_categories = len(ohe.categories_[0])
                self._metadata.append(
                    ColumnMetadata(
                        start_idx=cur_idx,
                        end_idx=cur_idx + n_categories,
                        discrete=True,
                        output_dim=n_categories,
                        model=ohe,
                        components=None,
                        name=column
                    )
                )
                cur_idx += n_categories
            else:
                bgm = BayesianGaussianMixture(
                    n_components=self._n_clusters,
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=0.001,
                    n_init=1
                )
                bgm.fit(column_data)
                components = bgm.weights_ > self._epsilon
                output_dim = components.sum() + 1
                self._metadata.append(
                    ColumnMetadata(
                        start_idx=cur_idx,
                        end_idx=cur_idx + output_dim,
                        discrete=False,
                        output_dim=output_dim,
                        model=bgm,
                        components=components,
                        name=column
                    )
                )
                cur_idx += output_dim
        self._output_dimensions = cur_idx
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms the passed DataFrame with the fitted data processor.

        Args:
            X (DataFrame):
                DataFrame used to fit the processor parameters.
                Should be aligned with the columns types defined in initialization.
        Returns:
            Processed version of the passed DataFrame.
        """
        if self._metadata is None:
            raise NotFittedError("This data processor has not yet been fitted.")
        
        transformed_data = []
        for col_md in self._metadata:
            column_data = X[[col_md.name]].values
            if col_md.discrete:
                ohe = col_md.model
                transformed_data.append(ohe.transform(column_data))
            else:
                bgm = col_md.model
                components = col_md.components

                means = bgm.means_.reshape((1, self._n_clusters))
                stds = np.sqrt(bgm.covariances_).reshape((1, self._n_clusters))
                features = (column_data - means) / (4 * stds)

                probabilities = bgm.predict_proba(column_data)
                n_opts = components.sum()
                features = features[:, components]
                probabilities = probabilities[:, components]

                opt_sel = np.zeros(len(column_data), dtype='int')
                for i in range(len(column_data)):
                    norm_probs = probabilities[i] + 1e-6
                    norm_probs = norm_probs / norm_probs.sum()
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=norm_probs)

                idx = np.arange((len(features)))
                features = features[idx, opt_sel].reshape([-1, 1])
                features = np.clip(features, -.99, .99)

                probs_onehot = np.zeros_like(probabilities)
                probs_onehot[np.arange(len(probabilities)), opt_sel] = 1
                transformed_data.append(
                    np.concatenate([features, probs_onehot], axis=1).astype(float))
                
        return np.concatenate(transformed_data, axis=1).astype(float)

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        """
        Reverts the data transformations on a passed DataFrame.

        Args:
            X (ndarray):
                Numpy array to be brought back to the original data format.
                Should share the schema of data transformed by this data processor.
                Can be used to revert transformations of training data or for synthetic samples.
        Returns:
            DataFrame with all performed transformations reverted.
        """
        if self._metadata is None:
            raise NotFittedError("This data processor has not yet been fitted.")

        transformed_data = []
        col_names = []
        for col_md in self._metadata:
            col_data = X[:, col_md.start_idx:col_md.end_idx]
            if col_md.discrete:
                inv_data = col_md.model.inverse_transform(col_data)
            else:
                mean = col_data[:, 0]
                variance = col_data[:, 1:]
                mean = np.clip(mean, -1, 1)

                v_t = np.ones((len(col_data), self._n_clusters)) * -100
                v_t[:, col_md.components] = variance
                variance = v_t
                means = col_md.model.means_.reshape([-1])
                stds = np.sqrt(col_md.model.covariances_).reshape([-1])

                p_argmax = np.argmax(variance, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]
                inv_data = mean * 4 * std_t + mean_t
            
            transformed_data.append(inv_data)
            col_names.append(col_md.name)

        transformed_data = np.column_stack(transformed_data)
        transformed_data = pd.DataFrame(transformed_data, columns=col_names).astype(self._dtypes)
        return transformed_data
