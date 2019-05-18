import json
import pandas as pd
import gzip
import os

import config

"""
Split dataset to train/eval files
Arguments:
    all_files: array of dataset files
"""
def split_dataset_into_two_groups(all_files):
    print("Started splitting dataset to A/B group")
    CONFIG = config.get_config()
    dtypes = config.get_types_of_attributes()
    df = pd.concat((pd.read_csv(f, sep=CONFIG['CSV_SEPARATOR'], compression='gzip', na_values=CONFIG['NA_VALUES'], dtype=dtypes) for f in all_files))
    group_a = df[df['player_group'] == 'A']
    group_b = df[df['player_group'] == 'B']
    group_a.to_csv("data/a_data.csv.gz", compression="gzip", sep=CONFIG['CSV_SEPARATOR'], index=False, na_rep="null")
    group_b.to_csv("data/b_data.csv.gz", compression="gzip", sep=CONFIG['CSV_SEPARATOR'], index=False, na_rep="null")
    print("Finished splitting dataset to A/B group")

"""
Split dataset to train/eval files
Arguments:
    all_files: array of dataset files
"""
def split_train_test(all_files):
    print("Started train test split")
    CONFIG = config.get_config()
    dtypes = config.get_types_of_attributes()
    df = pd.concat((pd.read_csv(f, sep=CONFIG['CSV_SEPARATOR'], compression='gzip', na_values=CONFIG['NA_VALUES'], dtype=dtypes) for f in all_files))
    train = df.sample(frac=0.7, random_state=CONFIG['RANDOM_SEED'])
    test = df.drop(train.index)
    train.to_csv("{}/train.csv.gz".format(CONFIG['DATASET_TRAIN'][0]), compression="gzip", sep=CONFIG['CSV_SEPARATOR'], index=False, na_rep="null")
    test.to_csv("{}/eval.csv.gz".format(CONFIG['DATASET_VAL'][0]), compression="gzip", sep=CONFIG['CSV_SEPARATOR'], index=False, na_rep="null")

    # with gzip.open("{}/train.csv.gz".format(CONFIG['DATASET_TRAIN']), "wb") as outfile:
    #        outfile.write(train_data.encode('utf-8'))
    # with gzip.open("{}/eval.csv.gz".format(CONFIG['DATASET_VAL']), "wb") as outfile:
    #        outfile.write(test_data.encode('utf-8'))
    print("Finished train test split")

"""
Calculate stats and saves in stats.json
Arguments:
    all_files: array of dataset files
"""
def calculate_stats(all_files):
    print("Started calculating stats")
    CONFIG = config.get_config()
    dtypes = config.get_types_of_attributes()
    df = pd.concat((pd.read_csv(f, sep=CONFIG['CSV_SEPARATOR'], compression='gzip', na_values=CONFIG['NA_VALUES'], dtype=dtypes) for f in all_files))
    stats_categorical = json.loads(df.describe(include='O').loc[[
        'count', 'unique'
    ]].to_json())
    stats_numeric = json.loads(df.describe().loc[[
        'count', 'mean', 'std', 'min', 'max'
    ]].to_json())

    weights = json.loads(df[CONFIG['DATASET_ID']].groupby([ df[label] for label in CONFIG['DATASET_WEIGHT_LABELS'] ]).agg(['count']).to_json())
    columns = df.columns.values

    try:
        with open(CONFIG['DATA_STATS_FILE'], 'w') as outfile:
            json.dump(obj={
                    'columns': {
                        'all': columns.tolist(),
                        'categorical': list(stats_categorical.keys()),
                        'numeric': list(stats_numeric.keys())
                    },
                    'stats': { **stats_numeric , **stats_categorical },
                    'weights': { **weights }
                }, fp=outfile, indent=4)
    except Exception as e:
        print(str(e))
    print("Finished calculating stats")


"""
Add extra columns to dataset and
create column,  activity_rate = 30 * ( (test_games_7d / 7) / (feature_1_games_30d / 30) )
                spend = 30 * (test_spend_7d / 7)
                weight
Arguments:
    all_files: array of dataset files
"""
def add_extra_columns_to_dataset(all_files):
    print("Started adding extra columns to dataset")
    CONFIG = config.get_config()
    dtypes = config.get_types_of_attributes()
    df = pd.concat((pd.read_csv(f, sep=CONFIG['CSV_SEPARATOR'], compression='gzip', na_values=CONFIG['NA_VALUES'], dtype=dtypes) for f in all_files))
    # print(df.describe())
    weights = json.loads(df[CONFIG['DATASET_ID']].groupby([ df[label] for label in CONFIG['DATASET_WEIGHT_LABELS'] ]).agg(['count']).to_json())
    total = float(df[CONFIG['DATASET_ID']].count())
    def get_weight(row):
        labels = CONFIG['DATASET_WEIGHT_LABELS']
        key = ','.join([ str(row[label]) for label in labels ])
        if len(labels) > 1:
            key = '[{}]'.format(key)
        freq = 1.0
        if key in weights['count']:
            freq = weights['count'][key]
#         prob = freq / total
#         target_prob = 1. / ( 2.0 ** len(labels) )
#         return target_prob / prob
        return total / ( ( 2.0 ** len(labels) ) * freq )

    def get_activity_rate(row):
        return ( (row['test_games_7d'] / 7.0) - (row['feature_1_games_30d'] / 30.0) )

    for f in all_files:
        df = pd.read_csv(f, sep=CONFIG['CSV_SEPARATOR'], compression='gzip', na_values=CONFIG['NA_VALUES'], dtype=dtypes)
        df['weight'] = 1.0
        # df['weight'] = df.apply (lambda row: get_weight(row), axis=1)
        df['activity_change'] = df.apply (lambda row: get_activity_rate(row), axis=1)
        df['spend'] = (df['test_spend_7d'] / 7.0)

        df.to_csv("{}".format(f), compression="gzip", sep=CONFIG['CSV_SEPARATOR'], index=False, na_rep="null")
        # new_data = gzip.compress(bytes(new_data, 'utf-8'))
        # with gzip.open("{}".format(CONFIG['WEIGHTED_DATASET_FILES']), "wb") as outfile:
	    #        outfile.write(new_data.encode('utf-8'))

    print("Finished adding extra columns to dataset")
