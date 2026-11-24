"""Nested host-grouped validation for candidate model-family/complexity selection.

This is intentionally separate from train_models.py. Run it before production
approval; it can be computationally expensive on a CPU-only runtime.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import f1_score, balanced_accuracy_score

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'AB_NYC_2019.csv'
SEED=42
WITHOUT=['neighbourhood_group','neighbourhood','latitude','longitude','price','minimum_nights','availability_365','calculated_host_listings_count']
WITH=WITHOUT+['number_of_reviews','reviews_per_month']
SCENARIOS={'without_reviews':WITHOUT,'with_reviews':WITH}

def make_pipeline(features, family):
    cats=[c for c in ['neighbourhood_group','neighbourhood'] if c in features]
    nums=[c for c in features if c not in cats]
    tx=[]
    ordinary=[c for c in nums if c!='reviews_per_month']
    if ordinary: tx.append(('num',SimpleImputer(strategy='median'),ordinary))
    if 'reviews_per_month' in nums: tx.append(('review',SimpleImputer(strategy='constant',fill_value=0),['reviews_per_month']))
    if cats: tx.append(('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore'))]),cats))
    prep=ColumnTransformer(tx,remainder='drop')
    if family=='random_forest':
        clf=RandomForestClassifier(random_state=SEED,n_jobs=1,class_weight='balanced_subsample')
    else:
        clf=HistGradientBoostingClassifier(random_state=SEED,max_iter=40,learning_rate=.08,class_weight='balanced',early_stopping=False)
    return Pipeline([('preprocessor',prep),('classifier',clf)])

def main():
    df=pd.read_csv(DATA); X=df[WITH]; y=df.room_type; groups=df.host_id
    tr,_=next(GroupShuffleSplit(n_splits=1,test_size=.30,random_state=SEED).split(X,y,groups))
    Xtr=X.iloc[tr]; ytr=y.iloc[tr]; gtr=groups.iloc[tr]
    all_rows=[]
    for scenario,features in SCENARIOS.items():
        outer=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED)
        for family in ['random_forest','hist_gradient_boosting']:
            grids=(
              [
                {'classifier__n_estimators':[30], 'classifier__max_depth':[18], 'classifier__min_samples_leaf':[2], 'classifier__max_features':['sqrt'], 'classifier__class_weight':['balanced_subsample']},
                {'classifier__n_estimators':[50], 'classifier__max_depth':[None], 'classifier__min_samples_leaf':[4], 'classifier__max_features':['sqrt'], 'classifier__class_weight':['balanced_subsample']},
              ]
              if family=='random_forest' else
              [
                {'classifier__max_iter':[40], 'classifier__max_leaf_nodes':[15], 'classifier__l2_regularization':[0.5], 'classifier__class_weight':['balanced']},
                {'classifier__max_iter':[60], 'classifier__max_leaf_nodes':[31], 'classifier__l2_regularization':[1.0], 'classifier__class_weight':['balanced']},
              ]
            )
            for fold,(oi,ov) in enumerate(outer.split(Xtr[features],ytr,groups=gtr),1):
                Xi=Xtr.iloc[oi][features]; yi=ytr.iloc[oi]; gi=gtr.iloc[oi]
                Xv=Xtr.iloc[ov][features]; yv=ytr.iloc[ov]
                inner=StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=SEED+fold)
                inner_splits=list(inner.split(Xi,yi,groups=gi))
                search=GridSearchCV(make_pipeline(features,family),param_grid=grids,scoring='f1_macro',cv=inner_splits,
                    n_jobs=1,refit=True,return_train_score=False,error_score='raise')
                search.fit(Xi,yi); pred=search.best_estimator_.predict(Xv)
                all_rows.append({'scenario':scenario,'family':family,'outer_fold':fold,
                    'inner_best_macro_f1':float(search.best_score_),
                    'outer_macro_f1':float(f1_score(yv,pred,average='macro',zero_division=0)),
                    'outer_balanced_accuracy':float(balanced_accuracy_score(yv,pred)),
                    'best_params':search.best_params_})
                print(all_rows[-1],flush=True)
    out=ROOT/'nested_validation_results.json'
    out.write_text(json.dumps(all_rows,indent=2),encoding='utf-8')
    print('Saved',out)

if __name__=='__main__': main()
