from monitoring import population_stability_index, retraining_recommendation
assert population_stability_index([1,2,3,4,5],[1,2,3,4,5]) < 1e-8
assert population_stability_index(['a','a','b'],['a','b','b']) >= 0
assert retraining_recommendation(max_feature_psi=.25)['action']=='HOLD_PROMOTION_AND_INVESTIGATE'
assert retraining_recommendation(current_macro_f1=.60,baseline_macro_f1=.70)['action']=='HOLD_PROMOTION_AND_INVESTIGATE'
print('Monitoring utility tests passed.')
