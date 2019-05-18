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
        'feature_1_games_30d': np.double,
        'feature_2' : np.string_,
        'feature_3' : np.double,
        'feature_4' : np.double,
        'feature_5' : np.double,
        'feature_6' : np.double,
        'feature_7' : np.string_,
        'feature_8' : np.string_,
        'feature_9' : np.string_,
        'feature_10' : np.double,
        'feature_11' : np.double,
        'feature_12' : np.double,
        'feature_13' : np.double,
        'feature_14' : np.string_,
        'feature_15' : np.double,
        'feature_16' : np.double,
        'feature_17' : np.double,
        'feature_18' : np.string_,
        'test_games_7d': np.double,
        'test_spend_7d': np.double,
        'player_group' : np.string_,
        'weight' : np.double,
        'activity_rate' : np.double,
        'spend' : np.double,        
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
