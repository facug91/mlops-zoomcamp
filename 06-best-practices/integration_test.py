import batch

import pandas as pd
import os

from deepdiff import DeepDiff


os.system('python batch.py 2023 1')

options = {
    'client_kwargs': {
        'endpoint_url': 'http://localhost:4566'
    }
}

df = pd.read_parquet('s3://nyc-duration/out/2023-01.parquet', storage_options=options)

actual_result = df.to_dict()

expected_result = {
    'ride_id': {
        0: '2023/01_0', 
        1: '2023/01_1'
    },
    'predicted_duration': {
        0: 23.197149, 
        1: 13.080101
    }
}

diff = DeepDiff(actual_result, expected_result, significant_digits=6)

assert 'type_changes' not in diff
assert 'values_changed' not in diff