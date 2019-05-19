import tensorflow as tf
import pandas as pd
import json
import math
import os
import numpy as np

import config

REMOVE_COLUMNS = []
LABEL_COLUMN = None
DATA_STATS = None
STATS_FILES = None

def set_stats_file(stat_files):
    global STATS_FILES
    STATS_FILES = stat_files

def set_stats(STATS=None):
    global DATA_STATS
    DATA_STATS = STATS

def get_stats():
    global DATA_STATS
    CONFIG = config.get_config()
    if DATA_STATS is None:
        with open(CONFIG['DATA_STATS_FILE'], 'r') as f:
            DATA_STATS = json.load(f)

    return DATA_STATS

def get_column_names():
    columns = get_stats()['columns']['all']

    return columns

def normalize(stats):
    CONFIG = config.get_config()
    # fn = lambda x: tf.where(tf.greater(tf.to_float(x), CONFIG["EPSILON"]), tf.log(tf.to_float(x)), tf.to_float(x))
    fn = lambda x: (tf.to_float(x) - stats['mean']) / (stats['std'] + CONFIG["EPSILON"])
#     fn = lambda x: (tf.to_float(x) - stats['min']) / (stats['max'] - stats['min'] + CONFIG["EPSILON"])
    return fn

def get_feature_columns3():
    stats = get_stats()
    CONFIG = config.get_config()
    numeric_features = []
    for key in stats['columns']['numeric']:
        if key in get_remove_columns() + get_label_column() + [ CONFIG['WEIGHT_COLUMN'] ]:
            continue
        numeric_features.append(
            tf.feature_column.numeric_column(key, normalizer_fn = normalize(stats['stats'][key]))
        )

    categorical_features = []
    for key in stats['columns']['categorical']:
        if key in get_remove_columns() + get_label_column() + [ CONFIG['WEIGHT_COLUMN'] ]:
            continue
        stat = stats['stats'][key]
        embedding_size = 6.0 * math.ceil(stat['unique']**0.25) if stat['unique'] > 25 else stat['unique']
        categorical_features.append(
            tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(key,
                                                 stat['unique_values']))
        )
    features = numeric_features + categorical_features

    return features

def get_feature_columns2():
    stats = get_stats()
    CONFIG = config.get_config()
    numeric_features = []
    for key in stats['columns']['numeric']:
        if key in get_remove_columns() + get_label_column() + [ CONFIG['WEIGHT_COLUMN'] ]:
            continue
        numeric_features.append(
            tf.feature_column.numeric_column(key, normalizer_fn = normalize(stats['stats'][key]))
        )

    categorical_features = []
    for key in stats['columns']['categorical']:
        if key in get_remove_columns() + get_label_column() + [ CONFIG['WEIGHT_COLUMN'] ]:
            continue
        stat = stats['stats'][key]
        embedding_size = 6.0 * math.ceil(stat['unique']**0.25) if stat['unique'] > 25 else stat['unique']
        categorical_features.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_hash_bucket(
                    key,
                    stat['unique'],
                    tf.string
                ),
                embedding_size + 1)
        )
    features = numeric_features + categorical_features

    return features

def get_feature_columns():
    prepare_csv_column_list()

    return get_feature_columns2()

def get_feature_columns_for_tree():
    prepare_csv_column_list()

    return get_feature_columns3()

def get_remove_columns():
    return REMOVE_COLUMNS

def set_remove_columns(remove_columns):
    global REMOVE_COLUMNS
    REMOVE_COLUMNS = remove_columns

def get_label_column():
    return LABEL_COLUMN

def set_label_column(label_column):
    global LABEL_COLUMN
    LABEL_COLUMN = label_column

""" Parse the CSV file of bidding data
Arguments:
    line: string, string of comma separated instance values
"""
def _parse_line(line):
#     print(line)
    # Decode the line into its fields
    CONFIG = config.get_config()
    features = tf.decode_csv(line, field_delim=CONFIG['CSV_SEPARATOR'], record_defaults=config.get_default_values_for_csv_columns(), na_value='null')
    features = dict(zip(get_column_names(), features))

    for column in get_remove_columns():
        features.pop(column)

    if get_label_column() is None:
        return features

    # Separate the label from the features
    labels = []
    for label in get_label_column():
        labels.append(features.pop(label))

    return features, labels

def prepare_csv_column_list():
    CONFIG = config.get_config()
    set_remove_columns(CONFIG['REMOVE_FEATURES'])
    set_label_column(CONFIG['LABELS'])

""" Eval input generator
Arguments:
    batch_size: number of instances to return
Returns:
    dataset tensor parsed from csv
"""
def train_input_fn(batch_size=1):
    CONFIG = config.get_config()
    filenames = []
    for path in CONFIG['DATASET_TRAIN']:
        new_files = []
        if os.path.isdir(path):
            new_files = [ os.path.join(path, p) for p in os.listdir(path) if p.endswith('.gz') ]
        else:
            new_files = [ path ]
        filenames = filenames + new_files
    prepare_csv_column_list()

    return csv_input_fn(filenames, batch_size, is_shuffle=True)

""" Eval input generator
Arguments:
    batch_size: number of instances to return
Returns:
    dataset tensor parsed from csv
"""
def validation_input_fn(batch_size=1):
    CONFIG = config.get_config()
    filenames = []
    for path in CONFIG['DATASET_VAL']:
        new_files = []
        if os.path.isdir(path):
            new_files = [ os.path.join(path, p) for p in os.listdir(path) if p.endswith('.gz') ]
        else:
            new_files = [ path ]
        filenames = filenames + new_files
    prepare_csv_column_list()

    return csv_input_fn(filenames, batch_size, is_shuffle=True)

""" Test input generator
Arguments:
    filenames: array of filename
Returns:
    dataset tensor parsed from csv
"""
def test_input_fn(filenames=None):
    CONFIG = config.get_config()
    if filenames is None:
        filenames = CONFIG['DATASET_TEST']
    prepare_csv_column_list()

    return csv_input_fn(filenames, batch_size=1, is_shuffle=False)

""" Return dataset in batches from a CSV file
Arguments:
    filenames: array of filename
    batch_size: number of instances to return
    is_shuffle: boolean if shuffle the dataset
Returns:
    dataset tensor parsed from csv
"""
def csv_input_fn(filenames, batch_size, is_shuffle=True):

    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    dataset = dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename, compression_type='GZIP').skip(1))
#     dataset = dataset.map(_parse_line)

    dataset = dataset.apply(tf.data.experimental.map_and_batch(_parse_line, batch_size, num_parallel_calls=4))

    # Shuffle, repeat, and batch the examples.
    if is_shuffle:
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(100, seed=42))

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels
