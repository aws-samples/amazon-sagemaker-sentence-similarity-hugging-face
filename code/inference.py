from sentence_transformers import SentenceTransformer, util
import csv
import os
import pandas as pd
from ipywidgets import widgets
import boto3
import json

bucket = 'sagemaker-us-east-1-827930657850'
s3key = 'sentencetransformer/input/train.csv'

def model_fn(model_dir):
    model = SentenceTransformer(model_dir)
    return model

def input_fn(data, context):
    print("************** input_fn *******************")
    AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
    )

    response = s3_client.get_object(Bucket=bucket, Key=s3key)
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    
    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        file = pd.read_csv(response.get("Body"), index_col=0)
    corpus = file['Description'].tolist()
    accident_level = file['Accident Level'].tolist()
    critical_risk = file['Critical Risk'].tolist()
    genre = file['Genre'].tolist()
    return [corpus, accident_level, critical_risk, genre, data]

def predict_fn(data,model):
    print("************** predict_fn *******************")
    corpus, accident_level, critical_risk, genre, sentence = data
    embeddings_corpus = model.encode(corpus)
    embeddings = model.encode(sentence)
    cosine_scores = util.pytorch_cos_sim(embeddings_corpus, embeddings)
    scores = cosine_scores.tolist()
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    data = {}
    data['result'] = []
    for i in indices:
        data['result'].append({
        'score': scores[i],
        'description': corpus[i],
        'accident-level': accident_level[i],
        'critical-risk': critical_risk[i],
        'genre': genre[i]
        })
    json_data = json.dumps(data)
    print("************** predict_fn end *******************")
    return json_data