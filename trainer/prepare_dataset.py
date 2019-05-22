""" Dataset transormation script """
import argparse
import os

import config
import dataset_helper

def transform(CONFIG):
    """ Transform dataset for training
    Args:
        CONFIG: configuration dictionary
    """
    dataset_helper.add_extra_columns_to_dataset(CONFIG['DATASET_FILES'])
    dataset_helper.split_train_test(CONFIG['DATASET_FILES'])
    path = CONFIG['DATASET_TRAIN'][0]
    train_files = [ os.path.join(path, p) for p in os.listdir(path) if p.endswith('.gz') ]
    dataset_helper.calculate_stats(train_files)

def main():
    os.environ['GA_CONFIG_PATH'] = 'config_spend_regressor_a.json'
    CONFIG = config.get_config()
    dataset_helper.split_dataset_into_two_groups(['data/data.csv.gz'])

    config_files = ['config_spend_regressor_a.json', 'config_spend_regressor_b.json']
    for file in config_files:
        os.environ['GA_CONFIG_PATH'] = file
        CONFIG = config.get_config()
        transform(CONFIG)

if __name__ == '__main__':
    main()
