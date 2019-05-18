import argparse

import requests

parser = argparse.ArgumentParser(description='Example api usage')
parser.add_argument('--objective', default='spend', type=str, help='objective name')
args = parser.parse_args()

API_ENDPOINT = "http://0.0.0.0:5000/group_allocator/predict?objective={}".format(args.objective)

data = {
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
}

r = requests.post(url=API_ENDPOINT, data=data)

print("{}".format(r.text))
