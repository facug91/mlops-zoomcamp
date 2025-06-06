import mlflow
import numpy as np
import pandas as pd
import pickle
import scipy

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from pathlib import Path
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import root_mean_squared_error

@task(name='Read Dataframe', retries=3, retry_delay_seconds=2, log_prints=True)
def read_dataframe(data_path: str) -> pd.DataFrame:
    """Read data into Dataframe"""
    df = pd.read_parquet(data_path)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df

@task(name='Add features', retries=3, retry_delay_seconds=2, log_prints=True)
def add_features(
    df: pd.DataFrame, dv: DictVectorizer = None
) -> tuple([scipy.sparse._csr.csr_matrix, np.ndarray, DictVectorizer]):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dicts = df[categorical + numerical].to_dict(orient='records')
    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    target = 'duration'
    y_train = df[target].values

    return X, y_train, dv


@task(name='Train model', retries=3, retry_delay_seconds=2, log_prints=True)
def train_model(X_train: scipy.sparse._csr.csr_matrix, y_train: np.ndarray, dv: DictVectorizer):
    """Train a linear regression model and write everything out"""

    with mlflow.start_run() as run:

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_train)

        rmse = root_mean_squared_error(y_train, y_pred)

        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('intercept_', lr.intercept_)

        Path('models').mkdir(exist_ok=True)
        with open('models/preprocessor.b', 'wb') as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact('models/preprocessor.b')

        mlflow.sklearn.log_model(lr, artifact_path='models')


@task(name='Register model', retries=3, retry_delay_seconds=2, log_prints=True)
def register_model():

    client = MlflowClient()

    experiment = client.get_experiment_by_name('nyc-taxi-experiment')
    
    run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.rmse ASC"]
    )[0]
    
    run_id = run.info.run_id
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name='trained-model'
    )


@flow(name='Main flow')
def main_flow(data_path: str = './data/yellow_tripdata_2023-03.parquet'):
    """The main training pipeline"""

    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('nyc-taxi-experiment')

    df_train = read_dataframe(data_path)

    X_train, y_train, dv = add_features(df_train)

    train_model(X_train, y_train, dv)

    register_model()


if __name__ == '__main__':
    main_flow()
