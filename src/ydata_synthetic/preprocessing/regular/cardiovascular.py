import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def transformations(data):
    categorical_features = ['gender', 'cardio', 'active', 'alco', 'smoke', 'gluc',
                'cholesterol']
    numerical_features = [ 'height', 'weight', 'ap_hi', 'ap_lo']

    numerical_transformer = Pipeline(steps=[
        ('onehot', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])
    
    processed_data = preprocessor.fit_transform(data)
    processed_data = pd.DataFrame.sparse.from_spmatrix(preprocessor.fit_transform(processed_data))
    return processed_data, preprocessor