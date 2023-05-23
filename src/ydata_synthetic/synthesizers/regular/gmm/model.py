"""
    GMM based synthetic data generation model
"""
from typing import List, Optional, Union

from joblib import dump, load
from tqdm import tqdm

from pandas import DataFrame
from numpy import (array, arange)

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from ydata_synthetic.synthesizers.base import BaseModel
from ydata_synthetic.preprocessing import RegularDataProcessor

class GMM(BaseModel):

    def __init__(self,
                 covariance_type:str="full",
                 random_state:int=0):
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.__MODEL__ = GaussianMixture(covariance_type=covariance_type,
                                         random_state=random_state)
        self.processor = RegularDataProcessor

    def __optimize(self, prep_data: array):
        """
        Auxiliary method to optimize the number of components to be considered for the Gaussian or Bayesian Mixture
        Returns:
            n_components (int): Optimal number of components calculated based on Silhouette score
        """
        c = arange(2, 40, 5)
        n_components=2
        max_silhouette=0
        for n in tqdm(c, desc="Hyperparameter search"):
            model = GaussianMixture(n, covariance_type=self.covariance_type, random_state=self.random_state)
            labels = model.fit_predict(prep_data)
            s = silhouette_score(prep_data, labels, metric='euclidean')
            if model.converged_:
                if max_silhouette < s:
                    n_components = n
                    max_silhouette=s
        return n_components

    def fit(self, data: Union[DataFrame, array],
                num_cols: Optional[List[str]] = None,
                cat_cols: Optional[List[str]] = None,):
        """
            ### Description:
            Trains and fit a synthesizer model to a given input dataset.

            ### Args:
            `data` (Union[DataFrame, array]): Training data
            `num_cols` (Optional[List[str]]) : List with the names of the categorical columns
            `cat_cols` (Optional[List[str]]): List of names of categorical columns

            ### Returns:
            **self:** *object*
                Fitted synthesizer
        """
        self.processor = RegularDataProcessor(num_cols=num_cols, cat_cols=cat_cols).fit(data)
        train_data = self.processor.transform(data)

        #optimize the n_components selection
        n_components = self.__optimize(train_data)

        self.__MODEL__.n_components=n_components
        #Fit the gaussian model
        self.__MODEL__.fit(train_data)

    def sample(self, n_samples: int):
        """
        ### Description:
        Generates samples from the trained synthesizer.

        ### Args:
        `n_samples` (int): Number of rows to generated.

        ### Returns:
        **synth_sample:** pandas.DataFrame, shape (n_samples, n_features)
            Returns the generated synthetic samples.
        """
        sample = self.__MODEL__.sample(n_samples=n_samples)[0]

        return self.processor.inverse_transform(sample)

    def save(self, path='str'):
        """
        Save a model as a pickle
        Args:
            path (str): The path where the model should be saved as pickle
        """
        try:
            with open(path, 'wb') as f:
                dump(self, f)
        except:
            raise Exception(f'The path {path} provided is not valid. Please validate your inputs')

    @classmethod
    def load(cls, path:str):
        """
        Load a trained synthesizer from a given path
        Returns:
            model (GMM): A trained GMM model
        """
        with open(path, 'rb') as f:
            model = load(f)
        return model
