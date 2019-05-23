import argparse
import pandas as pd
import requests
import numpy as np

def test(row):
    data = {
        "feature_2": row["feature_2"],
        "feature_3": row["feature_3"],
        "feature_4": row["feature_4"],
        "feature_5": row["feature_5"],
        "feature_6": row["feature_6"],
        "feature_7": row["feature_7"],
        "feature_8": row["feature_8"],
        "feature_9": row["feature_9"],
        "feature_10": row["feature_10"],
        "feature_11": row["feature_11"],
        "feature_12": row["feature_12"],
        "feature_13": row["feature_13"],
        "feature_14": row["feature_14"],
        "feature_15": row["feature_15"],
        "feature_16": row["feature_16"],
        "feature_17": row["feature_17"],
        "feature_18": row["feature_18"],
    }
    print(data)
    r = requests.post(url=API_ENDPOINT, data=data)
    spend = row['test_spend_7d'] / 7.0
    activity_change = ( ( row['feature_1_games_30d'] / 30.0) / ( ( row['test_games_7d'] + 1 ) / 7.0) )
    print("Actual group={}, spend={}, activity={}".format(row["player_group"], spend, activity_change))
    print("{}".format(r.text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example api usage')
    parser.add_argument('--objective', default='spend', type=str, help='objective name')
    args = parser.parse_args()
    dtypes = {
        'player_id' : np.string_,
        'feature_1_games_30d': np.double,
        'feature_2' : np.string_,
        'feature_3' : np.double,
        'feature_4' : np.double,
        'feature_5' : np.double,
        'feature_6' : np.double,
        'feature_7' : np.string_,
        'feature_8' : np.string_,
        'feature_9' : np.string_,
        'feature_10' : np.double,
        'feature_11' : np.double,
        'feature_12' : np.double,
        'feature_13' : np.double,
        'feature_14' : np.string_,
        'feature_15' : np.double,
        'feature_16' : np.double,
        'feature_17' : np.double,
        'feature_18' : np.string_,
        'test_games_7d': np.double,
        'test_spend_7d': np.double,
        'player_group' : np.string_,
        'weight' : np.double,
        'activity_change' : np.double,
        'spend' : np.double,
    }
    API_ENDPOINT = "http://0.0.0.0:5000/group_allocator/predict?objective=spend&objective=activity".format(args.objective)
    # 2578380,73.0,SU4,75.0,41.0,10.0,112.0,YW5kc,ZW4,eGlhb,0.0,1.0,0.0,41.0,null,4.0,50.0,44.0,KzA1OjMw,19.0,0.0,A,0.7106714423787595,0.2809523809523813,0.0
    df = pd.read_csv("data.csv.gz", sep=",", compression='gzip', na_values=["null", ""], dtype=dtypes)
    i = 0
    for index, row in df.iterrows():
        if i > 1:
            break
        test(row)
        i = i + 1
