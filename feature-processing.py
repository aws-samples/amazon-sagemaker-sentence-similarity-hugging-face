from __future__ import print_function

import argparse
import csv
import json
import os
import shutil
import sys
import time
from io import StringIO
from ast import literal_eval

import joblib
import numpy as np
import pandas as pd
from sagemaker_containers.beta.framework import (
    content_types,
    encoders,
    env,
    modules,
    transformer,
    worker,
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, OneHotEncoder, StandardScaler


label_column = "Critical Risk"


def _parse_args():

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    return parser.parse_known_args()

if __name__ == "__main__":

    args, _ = _parse_args()
    
    # process similar reports
    similar_reports = []
    with open(os.path.join(args.train, 'incident-batch.jsonl.out')) as f:
        for line in f:
            # converts jsonline array to normal array
            line = "[" + line.replace("[","").replace("]",",") + "]"
            similar_reports = literal_eval(line)
    batch_results = similar_reports[0].split('\"result\"')[1:]   
    similar_incident_features = np.empty([3, 3], dtype="S10")
    i=0
    for item in batch_results:
        top1 = item.split(", \"genre")[0]
        top1_risk = top1.split("critical-risk\": ")[1]
        top2 = item.split(", \"genre")[1]
        top2_risk = top2.split("critical-risk\": ")[1]
        top3 = item.split(", \"genre")[2]
        top3_risk = top3.split("critical-risk\": ")[1]
        print([top1_risk, top2_risk, top3_risk])
        similar_incident_features[i] = [top1_risk, top2_risk, top3_risk]
        i += 1
    
    
    # Load data
    df = pd.read_csv(os.path.join(args.train, 'train.csv'))
    df_test = df.head(3)
    df_test = df_test[['Industry Sector', 'Genre', 'Critical Risk']]
    df_test['top1_risk'] = similar_incident_features[0,:]
    df_test['top2_risk'] = similar_incident_features[1,:]
    df_test['top3_risk'] = similar_incident_features[2,:]

    print("## Processing complete. Saving model...")


    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, ['Industry Sector', 'Genre', 'Critical Risk']),
        ]
    )

    preprocessor.fit(df_test)

    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")


def input_fn(input_data, content_type):
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == "text/csv":
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None)
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == "text/csv":
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    features = model.transform(input_data)

    if label_column in input_data:
        # Return the label (as the first column) and the set of features.
        return np.insert(features, 0, input_data[label_column], axis=1)
    else:
        # Return only the set of features
        return features


def model_fn(model_dir):
    """Deserialize fitted model"""
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor