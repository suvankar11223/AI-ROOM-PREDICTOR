from pathlib import Path
import pandas as pd
from inference import load_model, predict, validate_input
ROOT=Path(__file__).resolve().parent
DATA=pd.read_csv(ROOT/'AB_NYC_2019.csv')
for scenario in ['without_reviews','with_reviews']:
    bundle=load_model(ROOT/'artifacts'/f'airbnb_room_type_{scenario}.joblib')
    cols=bundle['feature_columns']
    sample=DATA[cols].head(4).copy()
    out=predict(bundle,sample)
    assert len(out)==4 and out['predicted_room_type'].notna().all()
    prob_cols=[c for c in out if c.startswith('probability_')]
    assert (out[prob_cols].sum(axis=1)-1).abs().max()<1e-6
    # Unknown neighbourhoods are supported by handle_unknown='ignore'.
    unknown=sample.head(1).copy(); unknown['neighbourhood']='New Unseen Neighbourhood'
    assert len(predict(bundle,unknown))==1
    # Extra columns are ignored.
    extra=sample.head(1).copy(); extra['unrelated_extra_field']='ignored'
    assert len(predict(bundle,extra))==1
    # Missing required feature must fail clearly.
    try: predict(bundle,sample.drop(columns=[cols[0]]))
    except ValueError: pass
    else: raise AssertionError('Missing feature was not rejected.')
    # Invalid negative price must fail.
    invalid=sample.head(1).copy(); invalid['price']=-5
    try: predict(bundle,invalid)
    except ValueError: pass
    else: raise AssertionError('Negative price was not rejected.')
    # Invalid latitude must fail.
    invalid=sample.head(1).copy(); invalid['latitude']=120
    try: predict(bundle,invalid)
    except ValueError: pass
    else: raise AssertionError('Invalid latitude was not rejected.')
    if scenario=='with_reviews':
        # Structural missing review rate is accepted only when review count is zero.
        valid=sample.head(1).copy(); valid['number_of_reviews']=0; valid['reviews_per_month']=float('nan')
        assert len(predict(bundle,valid))==1
        invalid=sample.head(1).copy(); invalid['number_of_reviews']=5; invalid['reviews_per_month']=float('nan')
        try: predict(bundle,invalid)
        except ValueError: pass
        else: raise AssertionError('Inconsistent review-rate missingness was not rejected.')
    print(scenario, 'PASS')
print('All inference contract tests passed.')
