import json
import numpy as np
import os

"""
Get the configurations from config file as object
"""
def get_config():
    CONFIG = None
    with open(os.getenv('GA_CONFIG_PATH'), 'r') as f:
        CONFIG = json.load(f)
    return CONFIG

def get_types_of_attributes():
    return {
        'player_id' : np.string_,
        'feature_1_games_30d': np.double,       # 307219911482
        'feature_2' : np.string_,              # 3
        'feature_3' : np.double,                   # 01
        'feature_4' : np.double,           # 2
        'feature_5' : np.double,       # 12440
        'feature_6' : np.double,       # 8084
        'feature_7' : np.string_,    # 23678
        'feature_8' : np.string_,       # null
        'feature_9' : np.string_,      # null
        'feature_10' : np.double,   # 300x250
        'feature_11' : np.double,      # 0.707278907
        'feature_12' : np.double,     # null
        'feature_13' : np.double,         # 15166603496
        'feature_14' : np.string_,        # 4
        'feature_15' : np.double,     # 2
        'feature_16' : np.double,           # 19930
        'feature_17' : np.double,        # 100000
        'feature_18' : np.string_,
        'test_games_7d': np.double,
        'test_spend_7d': np.double,              # null
        'player_group' : np.string_,         # null
        'weight' : np.double,                 # 0
        'activity_rate' : np.double,                 # 0
        'spend' : np.double,                 # 0
    }

def get_default_values_for_csv_columns():
    default_value_for_dtypes = {
        np.string_: "",
        np.int_: 0,
        np.double: 0.0
    }
    types_of_attributes = get_types_of_attributes()
    default_values = []
    conf = get_config()
    for column_name, dtype in types_of_attributes.items():
        if column_name == conf['WEIGHT_COLUMN']:
            default_values.append(1.0)
        else:
            default_values.append(default_value_for_dtypes[dtype])

    return default_values

# print(get_default_values_for_csv_columns())
