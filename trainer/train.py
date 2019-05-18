from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import shutil
import functools
import os

import data
import config
import pandas as pd
import numpy as np

CONFIG = None

def get_weight_column():
    if CONFIG['WEIGHT_COLUMN']:
        return CONFIG['WEIGHT_COLUMN']
    return None

def get_dnn_regressor_model(hidden_units):
    runConfig = tf.estimator.RunConfig(
        save_checkpoints_steps = CONFIG['SAVE_CHECKPOINTS_STEPS'],
        save_summary_steps = CONFIG['SAVE_SUMMARY_STEPS'],
        tf_random_seed = CONFIG['RANDOM_SEED'])

    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir = CONFIG['MODEL_DIR'],
        weight_column = get_weight_column(),
        batch_norm = True,
        dnn_activation_fn=tf.nn.relu,
        dnn_hidden_units = hidden_units,
        dnn_dropout = CONFIG['DROPOUT'],
        dnn_feature_columns = data.get_feature_columns(),
        dnn_optimizer = lambda: tf.train.AdamOptimizer(
            # learning_rate=tf.train.exponential_decay(
            #         learning_rate=CONFIG['LEARNING_RATE'],
            #         global_step=tf.train.get_global_step(),
            #         decay_steps=CONFIG['LEARNING_RATE_DECAY_STEPS'],
            #         decay_rate=CONFIG['LEARNING_RATE_DECAY_RATE']
            #     )
            ),
        linear_feature_columns = data.get_feature_columns(),
        linear_optimizer = lambda: tf.train.AdamOptimizer(
            # learning_rate=tf.train.exponential_decay(
            #         learning_rate=CONFIG['LEARNING_RATE'],
            #         global_step=tf.train.get_global_step(),
            #         decay_steps=CONFIG['LEARNING_RATE_DECAY_STEPS'],
            #         decay_rate=CONFIG['LEARNING_RATE_DECAY_RATE']
            #     )
            ),
        config=runConfig)

    return estimator

""" Get the model definition """
def get_model():
    return get_dnn_regressor_model(CONFIG['NETWORK'])

def ensemble_architecture(result):
  """Extracts the ensemble architecture from evaluation results."""

  architecture = result["architecture/adanet/ensembles"]
  # The architecture is a serialized Summary proto for TensorBoard.
  summary_proto = tf.summary.Summary.FromString(architecture)
  return summary_proto.value[0].tensor.string_val[0]

""" Train the model """
def train_and_evaluate():
    estimator = get_model()
    serving_feature_spec = tf.feature_column.make_parse_example_spec(
      data.get_feature_columns())
    serving_input_receiver_fn = (tf.estimator.export.build_parsing_serving_input_receiver_fn(serving_feature_spec))

    exporter = tf.estimator.BestExporter(
      name=CONFIG['MODEL_NAME'],
      serving_input_receiver_fn=serving_input_receiver_fn,
      exports_to_keep=2
    )

    train_spec = tf.estimator.TrainSpec(
                       input_fn = lambda : data.train_input_fn(batch_size=CONFIG['BATCH_SIZE']),
                       max_steps = CONFIG['NUM_EPOCHS'])
    eval_spec = tf.estimator.EvalSpec(
                       input_fn = lambda : data.validation_input_fn(batch_size=CONFIG['BATCH_SIZE']),
                       steps = CONFIG['EVAL_STEPS'],
                       exporters=exporter,
                       start_delay_secs = 1, # start evaluating after N seconds
                       throttle_secs = 2)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print("Finished training")

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config.json', type=str, help='config path')
    parser.add_argument('--clean', default=0, type=int, help='Clean previously trained data')
    parser.add_argument('--is_test', default=0, type=int, help='Is Test')

    args = parser.parse_args(argv[1:])
    os.environ['GA_CONFIG_PATH'] = args.config_path
    global CONFIG
    CONFIG = config.get_config()
    if args.is_test > 0:
        pass
    else:
        tf.logging.set_verbosity(tf.logging.INFO)
        if args.clean > 0:
            shutil.rmtree(CONFIG['MODEL_DIR'], ignore_errors=True)
        train_and_evaluate()

if __name__ == '__main__':
    tf.app.run(main)
