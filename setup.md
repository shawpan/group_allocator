# Setup

requires `python3.7` and `docker`

```
git clone https://github.com/shawpan/group_allocator.git
cd group_allocator
bash create_env.sh
source group_allocator_environment/bin/activate
bash setup.sh
```

it will install the required packages and prepare for starting

# Start the app

```
bash start_app.sh
```

it will run tensorflow_model_server in docker, api server

**`objective` can be `spend` and/or `activity`**

Tensorflow model server:
[POST]
`http://localhost:8501/v1/models/predict_spend:regress`
`http://localhost:8501/v1/models/predict_activity_change:regress`
Api server
[POST]
`http://0.0.0.0:5000/group_allocator/predict?objective={objective1}&objective={objective2}`

with inputs as form data

```
"feature_2": "",
"feature_3": 0.0,
"feature_4": 0.0,
"feature_5": 0.0,
"feature_6": 0.0,
"feature_7": "",
"feature_8": "",
"feature_9": "",
"feature_10": 0.0,
"feature_11": 0.0,
"feature_12": 0.0,
"feature_13": 0.0,
"feature_14": "",
"feature_15": 0.0,
"feature_16": 0.0,
"feature_17": 0.0,
"feature_18": ""
```
