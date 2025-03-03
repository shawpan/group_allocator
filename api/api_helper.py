""" API helper methods """
import numpy as np
import os
import sys
import json
import requests

def set_config():
    """ Set configurations, raises error if config.json is not present
    """
    with open(os.path.join(sys.path[0], 'config.json'), 'r') as f:
        settings = json.dumps(json.load(f))
        os.environ['GA_API_CONFIG'] = settings

def get_config():
    """ Get configurations
    Returns:
        CONFIG dictionary
    """
    return json.loads(os.getenv('GA_API_CONFIG'))

def validate_input(request):
    """ Validate input
    Args:
        request: http request object
    Returns:
        a triple of (success, message, inputs)
    """
    inputs = []
    groups = ['A', 'B']
    for group in groups:
        input = {}
        for key, default_value in get_config()['input_keys'].items():
            input[key] = request.form.get(key, default_value, type=type(default_value))
        inputs.append(input)

    return True, 'Input is valid', inputs

def get_value(result):
    """ Get value from results object
    Args:
        result: prediction result object
    Returns:
        value from result
    """
    try:
        value = result['results'][0]
        return value
    except Exception as e:
        return None

def get_joined_result(result_a, result_b):
    """ Get predictions results joined in one
    Args:
        result_a: prediction result of model_a
        result_b: prediction result of model_b
    Returns:
        joined results
    """
    return {
        'results': [ get_value(result_a), get_value(result_b) ]
    }

def get_predictions(inputs, request):
    """ Get predictions for both groups for the different objectives
    Args:
        inputs: list of input object
        objectives: list of objective(string)
    Returns:
        Predictions
    """
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
            spend_result_a = make_request('predict_spend_a', payload)
            spend_result_b = make_request('predict_spend_b', payload)
            spend_result = get_joined_result(spend_result_a, spend_result_b)
        if objective == CONFIG['OBJECTIVES']['PREDICT_ACTIVITY']:
            activity_change_result_a = make_request('predict_activity_change_a', payload)
            activity_change_result_b = make_request('predict_activity_change_b', payload)
            activity_change_result = get_joined_result(activity_change_result_a, activity_change_result_b)

    return spend_result, activity_change_result

def get_best_group_single_objective(result):
    """ Get best group based on the predicted values
    Args:
        result: prediction result
    Returns:
        group (string)
    """
    group_max = None
    try:
        group_max = np.argmax(result["results"])
    except Exception as e:
        group_max = 0

    return group_max

def get_best_group(group_max_spend, activity_change_result):
    """ Get best group for activity based on the predicted values
    Args:
        group_max_spend: group index with maximum spend
        activity_change_result: prediction result for activity_change
    Returns:
        group (int)
    """
    try:
        CONFIG = get_config()
        activity_change_max_spend = activity_change_result["results"][group_max_spend]
        activity_change_other_group = activity_change_result["results"][1 - group_max_spend]
        # if activity decreased more than expected and less than the other group
        # return other group
        if activity_change_max_spend < CONFIG['ACTIVITY_DECREASE_THRESHOLD'] and activity_change_other_group > activity_change_max_spend:
            return 1 - group_max_spend
    except Exception as e:
        return group_max_spend

    return group_max_spend

def get_allocated_group(inputs, request):
    """ Allocate group based on the different objectives
    Args:
        inputs: list of input object
        objectives: list of objective(string)
    Returns:
        Allocated group (string)
    """
    groups = ["A", "B"]
    spend_result, activity_change_result = get_predictions(inputs, request)
    group_max_spend = get_best_group_single_objective(spend_result)
    group_max_activity = get_best_group_single_objective(activity_change_result)

    # if activity is not considered, return the group with highest spend
    if activity_change_result is None:
        return groups[group_max_spend], spend_result, activity_change_result
    # if spend is not considered, return the group with highest activity
    if spend_result is None:
        return groups[group_max_activity], spend_result, activity_change_result

    best_group_index = get_best_group(group_max_spend, activity_change_result)

    return groups[best_group_index], spend_result, activity_change_result

def make_request(model_name, payload):
    """ Make request to prediction api
    Args:
        model_name: name of the prediction model
        payload: inputs for the api endpoint
    Returns:
        predictions
    """
    try:
        r = requests.post(get_config()['prediction_api'].format(model_name), json=payload)
        pred = json.loads(r.content.decode('utf-8'))
        return pred
    except Exception as e:
        return None
