import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from pmlb import fetch_data

def transformations():
    data = fetch_data('adult')

    numerical_features = ['age', 'fnlwgt', 
                          'capital-gain', 'capital-loss',
                          'hours-per-week']
    numerical_transformer = Pipeline(steps=[
        ('onehot', StandardScaler())])

    categorical_features = ['workclass','education', 'marital-status', 
                            'occupation', 'relationship',
                            'race', 'sex']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    processed_data = pd.DataFrame.sparse.from_spmatrix(preprocessor.fit_transform(data))

    return data, processed_data, preprocessor


    