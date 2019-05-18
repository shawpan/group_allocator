import numpy as np
import os
import sys
import json
import requests

"""
Set configurations, raises error if config.json is not present
"""
def set_config():
    with open(os.path.join(sys.path[0], 'config.json'), 'r') as f:
        settings = json.dumps(json.load(f))
        os.environ['GA_API_CONFIG'] = settings

"""
Get configurations
Returns:
    CONFIG dictionary
"""
def get_config():
    return json.loads(os.getenv('GA_API_CONFIG'))

"""
Validate input
Args:
    request: http request object
Returns:
    a triple of (success, message, inputs)
"""
def validate_input(request):
    inputs = []
    groups = ['A', 'B']
    for group in groups:
        input = {}
        for key, default_value in get_config()['input_keys'].items():
            input[key] = request.form.get(key, default_value)
            if str(input[key]).isnumeric():
                input[key] = float(input[key])
        input['player_group'] = group
        inputs.append(input)
    return True, 'Input is valid', inputs

"""
Get predictions for both groups for the different objectives
Arguments:
    inputs: list of input object
    objectives: list of objective(string)
Returns:
    Predictions
"""
def get_predictions(inputs, request):
    payload = {
        "examples": inputs
    }
    spend_result = activity_change_result = None
    CONFIG = get_config()
    objectives = request.args.getlist('objective')
    if len(objectives) == 0:
        objectives.append(CONFIG['OBJECTIVES']['PREDICT_SPEND'])

    for objective in objectives:
        if objective == CONFIG['OBJECTIVES']['PREDICT_SPEND']:
            spend_result = make_request('predict_spend', payload)
        if objective == CONFIG['OBJECTIVES']['PREDICT_ACTIVITY']:
            activity_change_result = make_request('predict_activity_change', payload)

    return spend_result, activity_change_result

"""
Get best group based on the predicted values
Arguments:
    result: prediction result
Returns:
    group (string)
"""
def get_best_group_single_objective(result):
    group_max = None
    try:
        group_max = np.argmax(result["results"])
    except Exception as e:
        group_max = 0

    return group_max

"""
Get best group for activity based on the predicted values
Arguments:
    result: prediction result
Returns:
    group (string)
"""
def get_best_group(group_max_spend, result):
    try:
        CONFIG = get_config()
        activity_change_max_spend = result["results"][group_max_spend]
        activity_change_other_group = result["results"][1 - group_max_spend]
        # if activity decreased more than expected and less than the other group
        # return other group
        if activity_change_max_spend < CONFIG['ACTIVITY_DECREASE_THRESHOLD'] and activity_change_other_group > activity_change_max_spend:
            return 1 - group_max_spend
    except Exception as e:
        return group_max_spend

    return group_max_spend

"""
Allocate group based on the different objectives
Arguments:
    inputs: list of input object
    objectives: list of objective(string)
Returns:
    Allocated group (string)
"""
def get_allocated_group(inputs, request):
    groups = ["A", "B"]
    spend_result, activity_change_result = get_predictions(inputs, request)
    group_max_spend = get_best_group_single_objective(spend_result)
    group_max_activity = get_best_group_single_objective(activity_change_result)

    # if activity is not considered, return the group with highest spend
    if activity_change_result is None:
        return groups[group_max_spend]
    # if spend is not considered, return the group with highest activity
    if spend_result is None:
        return groups[group_max_activity]

    return get_best_group(group_max_spend, activity_change_result)

"""
Make request to prediction api
Aarguments:
    model_name: name of the prediction model
    payload: inputs for the api endpoint
Returns:
    predictions
"""
def make_request(model_name, payload):
    try:
        r = requests.post(config.get_config()['prediction_api'].format(model_name), json=payload)
        pred = json.loads(r.content.decode('utf-8'))
        return pred
    except Exception as e:
        return None
