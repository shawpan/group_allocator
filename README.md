# Set up
The procedure to run the application is described in [setup.md] https://github.com/shawpan/group_allocator/blob/master/setup.md

# Task Description

**Background:** Game designers has created two different versions of live-ops (noted as version "A" and "B") for a certain mobile game; but they are not sure which is better.

**Overall objective:** Assign an individual game player to a group (A/B) in order to maximize his/her future activities (e.g. amount of game-play and/or spend).

**Dataset collection:** an A/B test has been carried out on two groups (each group gets exposed to design A or B) of players selected randomly; and a dataset is gathered.

# Basic flow of the algorithm
[algoflow]: ./trainer/basic.png "Algorithm Flow"
![FlowChart][algoflow]

# Description of the steps

1. Dataset is splitted into two groups, Dataset A with `player_group=A` and Dataset B with `player_group=B`
2. Data_A and Data_B are transormed to have two new columns `spend` and `activity_change`.

   ```
   spend = test_spend_7d / 7.0
   activity_change =  (feature_1_games_30d / 30.0) / ( (test_games_7d + 1) / 7.0)
   ```

   Both Data_A and Data_B are splitted into train and eval file having 70/30 ratio of random instances.
   Mean, std, uniques etc are calculated for each column for two train datasets and saved into two files for future use.
3. For features `feature_2` - `feature_18` are used. Numeric columns are normalized using `(x-mean)/(std + epsilon)` and categorical columns are transformed to hash buckets and then into embedding columns. bucket size is the number of uniques of the feature and embedding size = `6 * number of unique^0.25` if `number of unique > 25` otherwise `number of unique`. `trainer/plot_data.ipynb` notebook draws basic joint distribution charts.

4. Models are Neural Networks having different hyperparameters i.e, number of layers, number of nodes in each layer, learning rate etc. for model_spend and model_activity also dependiing on dataset A and B.
`tf.estimator.BaseLineRegressor` is used as a baseline.
Tried
`tf.estimator.DNNRegressor`,
`tf.estimator.LinearRegressor`,
`tf.estimator.BoostedTreesRegressor`,
`tf.estimator.DNNLinearCombinedRegressor` (combines linear + dnn)
at the end sticking to `tf.estimator.DNNRegressor` for the best performance.
     - `trainer/train.py` has the training models
     - `trainer/data.py` has the dataset api methods
     - `trainer/config_*.json` (4 files) have the configurations of hyperparameters.

5. Evaluation metrics of each model is given in [metrics.md] https://github.com/shawpan/group_allocator/blob/master/metrics.md

# Implementation
1. Models are built and trained using Tensorflow and served using Tensorflow Server model docker image. `/trainer` directory has the necessary codes to train the models.
To train models, e.g,

Four models need to be trained

1. spend prediction on dataset A
2. spend prediction on dataset B
3. activity prediction on dataset A
4. activity prediction on dataset B

Train models since beginning

```
python train.py --config_path=config_spend_regressor_a.json clean=1
python train.py --config_path=config_spend_regressor_b.json clean=1
python train.py --config_path=config_activity_regressor_a.json clean=1
python train.py --config_path=config_activity_regressor_b.json clean=1
```

Train models from last checkpoint

```
python train.py --config_path=config_spend_regressor_a.json
python train.py --config_path=config_spend_regressor_b.json
python train.py --config_path=config_activity_regressor_a.json
python train.py --config_path=config_activity_regressor_b.json
```

API endpoint Examples
```
[POST] http://localhost:8501/v1/models/predict_spend_a:regress
[POST] http://localhost:8501/v1/models/predict_spend_b:regress
[POST] http://localhost:8501/v1/models/predict_activity_change_a:regress
[POST] http://localhost:8501/v1/models/predict_activity_change_b:regress

with payload
{
	"examples": [
		{
        "feature_2": "VFI",
		    "feature_3": 135.0,
		    "feature_4": 14.0,
		    "feature_5": 9.0,
		    "feature_6": 52.0,
		    "feature_7": "YW5kc",
		    "feature_8": "dHI",
		    "feature_9": "c2Ftc",
		    "feature_10": 2953383936.0,
		    "feature_11": 1.0,
		    "feature_12": 0.0,
		    "feature_13": 811.0,
		    "feature_14": "",
		    "feature_15": 2.0,
		    "feature_16": 3.0,
		    "feature_17": 0.0,
		    "feature_18": "KzAzOjAw"
		}
	]
}
sample response
{
    "results": [1.99398]
}
```
2. API end point is implemented using flask api tool. Codes are in `/api` directory
example
```
[POST] http://0.0.0.0:5000/group_allocator/predict?objective=spend&objective=activity

[POST] http://0.0.0.0:5000/group_allocator/predict?objective=spend

[POST] http://0.0.0.0:5000/group_allocator/predict?objective=activity

with payload
{
  'feature_2': 'R1Q',
  'feature_3': 63.0,
  'feature_4': 42.0,
  'feature_5': 8.0,
  'feature_6': 106.0,
  'feature_7': 'YW5kc',
  'feature_8': 'ZXM',
  'feature_9': 'c2Ftc',
  'feature_10': 0.0,
  'feature_11': 1.0,
  'feature_12': 0.0,
  'feature_13': 88.0,
  'feature_14': nan,
  'feature_15': 4.0,
  'feature_16': 50.0,
  'feature_17': 24.0,
  'feature_18': 'LTA2OjAw'
}
sample response
{
  "result": {
    "activityChange": {
      "results": [
        1.9473,
        1.81868
      ]
    },
    "group": "A",
    "spendResult": {
      "results": [
        6.4428,
        0.395823
      ]
    }
  }
}
```

# Remarks

Performance can be improved further because training loss decreased but the evaluation loss decay is too low throughout the training process. There is a clear scope of generalization improvement since the model is slightly overfitted (visible gap between training and evaluation loss ). Extensive feature engineering + hyperparameter tuning must improve the model further. To start with learning_rate can be decreased carefully and gradually before applying dropouts and a more narrow and deep network. 
