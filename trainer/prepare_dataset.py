import argparse
import os

import config
import dataset_helper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config.json', type=str, help='config path')

    args = parser.parse_args()
    os.environ['GA_CONFIG_PATH'] = args.config_path
    CONFIG = config.get_config()
    dataset_helper.add_extra_columns_to_dataset(CONFIG['DATASET_FILES'])
    dataset_helper.calculate_stats(CONFIG['WEIGHTED_DATASET_FILES'])
    dataset_helper.split_train_test(CONFIG['WEIGHTED_DATASET_FILES'])

if __name__ == '__main__':
    main()
