import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import root_mean_squared_error


import logging 

import logging
import pandas as pd
from typing import Tuple, Optional
from sklearn.feature_extraction import DictVectorizer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

def download_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        logger.info("Downloading data")
        green_jan = pd.DataFrame(pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet'))
        green_feb = pd.DataFrame(pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-02.parquet'))
        green_mar = pd.DataFrame(pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-03.parquet'))
        logger.info("Data Downloaded")
        return green_jan, green_feb, green_mar
    except Exception as e:
        logger.exception("Failed Downloading Data")
        raise

def preprocess(df: pd.DataFrame, dv: Optional[DictVectorizer] = None, fit_dv: bool = False) -> Tuple[np.ndarray, Optional[DictVectorizer]]:
    try:
        logger.info("Processing data")
        
        df = df.copy()  
        df['duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).apply(lambda td: td.total_seconds() / 60)
        df = df[(df.duration >= 1) & (df.duration <= 60)]

        
        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)

        
        dicts = df[categorical].to_dict(orient='records')
        
        
        if fit_dv:
            dv = DictVectorizer()
            X = dv.fit_transform(dicts)
        else:
            if dv is None:
                raise ValueError("DictVectorizer must be provided when fit_dv=False")
            X = dv.transform(dicts)
        
        logger.info("Data Processed")
        return X, dv
    except Exception as e:
        logger.exception("Preprocessing Failed")
        raise

def extract_x_y(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    try:
        logger.info("Setting XY")
        target = 'duration'
        y_train = np.array(df_train[target].values, dtype=np.float64)
        y_val = np.array(df_val[target].values, dtype=np.float64)
        y_test = np.array(df_test[target].values, dtype=np.float64)
        logger.info("XY Set")
        return target, y_train, y_val, y_test
    except Exception as e:
        logger.exception("Could not extract XY")
        raise

def fit_dict(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DictVectorizer]:
    try:
        logger.info("Vectorizing Dict")
        
        X_train, dv = preprocess(df_train, fit_dv=True)
        X_val, _ = preprocess(df_val, dv, fit_dv=False)
        X_test, _ = preprocess(df_test, dv, fit_dv=False)
        logger.info("Vectorization Complete")
        return X_train, X_val, X_test, dv
    except Exception as e:
        logger.exception("Vectorization Failed")
        raise

def main():
    try:
        
        df_train, df_val, df_test = download_data()
        X_train, X_val, X_test, dv = fit_dict(df_train, df_val, df_test)
        target, y_train, y_val, y_test = extract_x_y(df_train, df_val, df_test)
        logger.info("Pipeline completed successfully")
        return X_train, X_val, X_test, y_train, y_val, y_test, dv, target
    except Exception as e:
        logger.exception("Pipeline failed")
        raise

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, dv, target = main()