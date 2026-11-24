"""Rebuild both model bundles from AB_NYC_2019.csv.

This reproduces the current candidate model recipe. It does not complete the
separate nested-CV or newer-snapshot validation gates documented in README.md.
"""
from pathlib import Path
import hashlib, json, platform
from datetime import datetime, timezone
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'AB_NYC_2019.csv'
ART=ROOT/'artifacts'; ART.mkdir(exist_ok=True)
SEED=42
WITHOUT=['neighbourhood_group','neighbourhood','latitude','longitude','price','minimum_nights','availability_365','calculated_host_listings_count']
WITH=WITHOUT+['number_of_reviews','reviews_per_month']
SCENARIOS={'without_reviews':WITHOUT,'with_reviews':WITH}
CLASSES=['Entire home/apt','Private room','Shared room']

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def make_pipeline(features):
    cats=[c for c in ['neighbourhood_group','neighbourhood'] if c in features]
    nums=[c for c in features if c not in cats]
    transformers=[]
    ordinary=[c for c in nums if c!='reviews_per_month']
    if ordinary: transformers.append(('numeric',SimpleImputer(strategy='median'),ordinary))
    if 'reviews_per_month' in nums: transformers.append(('review_rate',SimpleImputer(strategy='constant',fill_value=0),['reviews_per_month']))
    if cats:
        transformers.append(('categorical',Pipeline([
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))
        ]),cats))
    pre=ColumnTransformer(transformers,remainder='drop')
    clf=HistGradientBoostingClassifier(max_iter=60,learning_rate=.08,max_leaf_nodes=31,
        l2_regularization=1.0,class_weight='balanced',early_stopping=False,random_state=SEED)
    return Pipeline([('preprocessor',pre),('classifier',clf)])

df=pd.read_csv(DATA)
assert df['id'].is_unique and not df.duplicated().any()
assert (df['reviews_per_month'].isna()==df['number_of_reviews'].eq(0)).all()
X=df[WITH]; y=df['room_type']; groups=df['host_id']
train_idx,test_idx=next(GroupShuffleSplit(n_splits=1,test_size=.30,random_state=SEED).split(X,y,groups))
X_train=X.iloc[train_idx]; y_train=y.iloc[train_idx]; groups_train=groups.iloc[train_idx]
X_test=X.iloc[test_idx]; y_test=y.iloc[test_idx]; groups_test=groups.iloc[test_idx]
assert set(groups_train).isdisjoint(set(groups_test))

for scenario,features in SCENARIOS.items():
    cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED+50)
    splits=list(cv.split(X_train[features],y_train,groups=groups_train))
    estimator=CalibratedClassifierCV(estimator=make_pipeline(features),method='sigmoid',cv=splits,ensemble=True)
    estimator.fit(X_train[features],y_train)
    bundle={
        'bundle_format':'nyc_airbnb_room_type_model_bundle_v3','model_version':'3.0.0','scenario':scenario,
        'estimator':estimator,'feature_columns':features,'classes':estimator.classes_.tolist(),
        'decision_policy':{'type':'argmax_calibrated_probability'},
        'input_schema':{'required_features':features,
            'categorical_features':[c for c in features if c in ['neighbourhood_group','neighbourhood']],
            'numeric_features':[c for c in features if c not in ['neighbourhood_group','neighbourhood']],
            'constraints':{'latitude':[-90,90],'longitude':[-180,180],'price':[0,None],
                'minimum_nights':[0,None],'availability_365':[0,365],
                'calculated_host_listings_count':[0,None],'number_of_reviews':[0,None],'reviews_per_month':[0,None]}},
        'prediction_scope':'Cross-sectional classification of an NYC listing snapshot. The no-review variant excludes review-history fields but is not validated as a listing-creation forecast.',
        'training_data':{'file':'AB_NYC_2019.csv','sha256':sha256(DATA),'rows':len(df),'snapshot_year':2019},
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'environment':{'python':platform.python_version(),'scikit_learn':sklearn.__version__,
            'numpy':np.__version__,'pandas':pd.__version__,'joblib':joblib.__version__}}
    path=ART/f'airbnb_room_type_{scenario}.joblib'
    joblib.dump(bundle,path,compress=3)
    loaded=joblib.load(path)
    assert np.allclose(estimator.predict_proba(X_test[features].head(20)),loaded['estimator'].predict_proba(X_test[features].head(20)))
    print(f'{scenario}: saved {path.name}; {path.stat().st_size:,} bytes')
print('Both model bundles rebuilt. Run the external and nested validation gates before production approval.')
