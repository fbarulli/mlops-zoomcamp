import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)


































def main():
    try:
        logger.info("Starting MLFlow experiment")
        
        logger.info("Experiment Complete")
    except Exception as e:
        logger.exception("Experiment Failed")
        raise
    
    
    
if __name__ == "__main__":
    main()