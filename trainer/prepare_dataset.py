import argparse
import os

import config
import dataset_helper

def transform(CONFIG):
    dataset_helper.add_extra_columns_to_dataset(CONFIG['DATASET_FILES'])
    dataset_helper.calculate_stats(CONFIG['DATASET_FILES'])
    dataset_helper.split_train_test(CONFIG['DATASET_FILES'])

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
