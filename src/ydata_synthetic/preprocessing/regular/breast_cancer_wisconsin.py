import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from pmlb import fetch_data

def transformations(auto=True):
    if auto:
        data = fetch_data('breast_cancer_wisconsin')
    else:
        data = fetch_data('breast_cancer_wisconsin')
        
    scaler = StandardScaler()
    processed_data = scaler.fit_transform(data)
    processed_data = pd.DataFrame(processed_data)
    
    return data, processed_data, scaler


if __name__ == '__main__':
    
    data = transformations(auto=True)
    
    print(data)
    