from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
import optuna.integration.lightgbm as lgb

iris = load_iris()
X_train = iris.data
y_train = iris.target
lgb_train = lgb.Dataset(X_train, y_train)

FOLD = 5
NUM_ROUND = 10000
VERBOSE_EVAL = 1
NUM_CLASS = y_train.max() + 1

folds = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=42)

lgb_train = lgb.Dataset(X_train, y_train)

params = {'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': NUM_CLASS,
        'boosting_type': 'gbdt',
        'deterministic':True,
        'force_row_wise':True
        } 
        
tuner = lgb.LightGBMTunerCV(params, lgb_train, 
    verbose_eval=100, num_boost_round=NUM_ROUND, early_stopping_rounds=100, 
    folds=folds)
    
tuner.run()

best_params = tuner.best_params
print("  Params: ")
for key, value in best_params.items():
    print("    {}: {}".format(key, value))
