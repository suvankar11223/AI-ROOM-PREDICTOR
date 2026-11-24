"""Evaluate a newer labelled Inside Airbnb NYC listings CSV against a saved bundle.

Example:
  python external_validation.py --csv ./new_york_city_listings.csv \
    --model artifacts/airbnb_room_type_without_reviews.joblib \
    --snapshot-date 2026-09-14 --output external_validation.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from inference import load_model, predict

ALIASES = {
    "neighbourhood_group": ["neighbourhood_group", "neighbourhood_group_cleansed"],
    "neighbourhood": ["neighbourhood", "neighbourhood_cleansed"],
    "latitude": ["latitude"], "longitude": ["longitude"], "price": ["price"],
    "minimum_nights": ["minimum_nights"], "availability_365": ["availability_365"],
    "calculated_host_listings_count": ["calculated_host_listings_count", "host_listings_count"],
    "number_of_reviews": ["number_of_reviews"], "reviews_per_month": ["reviews_per_month"],
}

def normalize_external(frame, features):
    mapped = pd.DataFrame(index=frame.index)
    missing = []
    for feature in features:
        source = next((c for c in ALIASES.get(feature, [feature]) if c in frame.columns), None)
        if source is None:
            missing.append(feature)
        else:
            mapped[feature] = frame[source]
    if missing:
        raise ValueError(f"New snapshot is missing required compatible fields: {missing}. Do not fabricate replacements.")
    if "price" in mapped:
        mapped["price"] = (mapped["price"].astype(str).str.replace(r"[$,]", "", regex=True)
                           .replace({"nan": None, "": None}).astype(float))
    return mapped

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--snapshot-date',required=True)
    parser.add_argument('--output',default='external_validation.json')
    args=parser.parse_args()
    frame=pd.read_csv(args.csv)
    if 'room_type' not in frame:
        raise ValueError("The external snapshot has no room_type labels; labelled external validation is impossible.")
    bundle=load_model(args.model)
    X=normalize_external(frame,bundle['feature_columns'])
    valid=frame['room_type'].notna()
    X=X.loc[valid]; y=frame.loc[valid,'room_type']
    result=predict(bundle,X)
    yhat=result['predicted_room_type']
    labels=bundle['classes']
    report={"snapshot_date":args.snapshot_date,"rows_evaluated":int(len(y)),
      "unique_hosts":int(frame.loc[valid,'host_id'].nunique()) if 'host_id' in frame else None,
      "accuracy":float(accuracy_score(y,yhat)),"macro_f1":float(f1_score(y,yhat,labels=labels,average='macro',zero_division=0)),
      "weighted_f1":float(f1_score(y,yhat,average='weighted',zero_division=0)),
      "balanced_accuracy":float(balanced_accuracy_score(y,yhat)),
      "classification_report":classification_report(y,yhat,labels=labels,output_dict=True,zero_division=0),
      "confusion_matrix":confusion_matrix(y,yhat,labels=labels).tolist(),
      "model_version":bundle['model_version'],"scenario":bundle['scenario'],
      "warning":"A later snapshot evaluates temporal shift, but may include previously seen hosts. Report host overlap separately before calling it unseen-host validation."}
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
