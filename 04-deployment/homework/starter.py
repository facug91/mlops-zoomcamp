#!/usr/bin/env python
# coding: utf-8

import pickle
import os
import argparse
import pandas as pd


def run(taxi_type: str, year: int, month: int):

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    def read_data(filename):
        print(f'Downloading parquet file')
        df = pd.read_parquet(filename)
        
        print(f'Prepring dataframe')
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        
        return df

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet')

    print(f'Processing dataframe')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f'Mean {y_pred.mean()}')
    print(f'Std {y_pred.std()}')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    print(f'Building result dataframe')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    # df_result['tpep_pickup_datetime'] = df['tpep_pickup_datetime']
    # df_result['PULocationID'] = df['PULocationID']
    # df_result['DOLocationID'] = df['DOLocationID']
    # df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    # df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']

    os.makedirs('output', exist_ok=True)
    output_file = f'output/yellow_{year:04d}-{month:02d}_output.parquet'
    print(f'Writing results to {output_file}')

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--taxi_type', help='Taxi color to process')
    parser.add_argument('--year', help='Year to process', type=int)
    parser.add_argument('--month', help='Month to process', type=int)

    args = parser.parse_args()

    run(args.taxi_type, args.year, args.month)