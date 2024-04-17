
import pandas as pd
import numpy as np
import argparse
import os
from ast import literal_eval
from sklearn.preprocessing import OrdinalEncoder

def _parse_args():

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/report/')
    parser.add_argument('--filename', type=str, default='train.csv')
    parser.add_argument('--filepath_similar_report', type=str, default='/opt/ml/processing/input/similar_report/')
    parser.add_argument('--filename_similar_report', type=str, default='incident-batch.jsonl.out')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')

    return parser.parse_known_args()

if __name__=="__main__":
    # Process arguments
    args, _ = _parse_args()
    
    # process similar reports
    similar_reports = []
    with open(os.path.join(args.filepath_similar_report, args.filename_similar_report)) as f:
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
    df = pd.read_csv(os.path.join(args.filepath, args.filename))
    df_test = df.head(3)
    df_test['top1_risk'] = similar_incident_features[0,:]
    df_test['top2_risk'] = similar_incident_features[1,:]
    df_test['top3_risk'] = similar_incident_features[2,:]
    df_test.to_csv(os.path.join(args.outputpath, 'test/test-processing.csv'), index=False, header=False)
    df_test.to_csv(os.path.join(args.outputpath, 'train/test-processing.csv'), index=False, header=False)
    df_test.to_csv(os.path.join(args.outputpath, 'validation/test-processing.csv'), index=False, header=False)

    print("## Processing complete. Exiting.")
    print(df_test)
