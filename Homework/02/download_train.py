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

logger = logging.getLogger(__name__)

def download_data():
    logger.info("Downloading data")
    green_jan = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet')
    green_feb = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-02.parquet')
    green_mar = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-03.parquet')
    logger.info("Data Downloaded")
    return green_jan, green_feb, green_mar

def preprocess(df):
    logger.info("Processing data")
    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    logger.info("Data Processed")
    return df


def extract_x_y(df):
    logger.info("Setting XY")
    target = 'duration'
    y_train = df[target].values
    y_val = df[target].values
    y_test = df[target].values
    logger.info("XY Set")
    return target, y_train, y_val, y_test


def fit_dict(df_train, df_val, df_test):
    logger.info("Vectorizing Dict")
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    
    
    
    