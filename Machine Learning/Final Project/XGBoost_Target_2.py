"""
XGBoost Project for School

Script below is specifically for target2 using a XGBoost technique.

@author: Thomas Gow
"""
#%% Library
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

#%% Data
train = pd.read_csv(r"/Users/thomasgow/Documents/IAA/Machine Learning/Competition/MLProject_train.csv", engine='python')
valid = pd.read_csv(r"/Users/thomasgow/Documents/IAA/Machine Learning/Competition/MLProject_valid.csv")

#%% Drop target1 column
train = train.drop(columns=['target1'])
valid = valid.drop(columns=['target1'])

#%% Create clean data and appropriate dataframes
train.dropna(inplace=True)
valid.dropna(inplace=True)

# Create sample for easier tuning
sample = train.sample(n = 10000)

# Create appropriate dataframes
X_sample = sample.drop(columns=['target2'])
X_train = train.drop(columns=['target2'])
X_test = valid.drop(columns=['target2'])
y_sample = sample['target2']
y_train = train['target2']
y_test = valid['target2']

#%% DMatrices have better performance with XGBoost
dtrain = xgb.DMatrix(X_sample, label=y_sample)
dtest = xgb.DMatrix(X_test, label=y_test)

#%% Create parameters
params = {
    # Parameters that are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'binary:hinge',
    'eval_metric':'auc'
}
num_boost_round = 999

#%% Train the first model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)

#%%
print("Best MAE: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))

#%% Conduct CV with error evaluation
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'error'},
    early_stopping_rounds=10
)
cv_results

#%% Print minimum test error
cv_results['test-error-mean'].min()

#%% Conducting same analysis with "AUC" score
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'auc'},
    early_stopping_rounds=10
)
cv_results

#%% Print auc best value
cv_results['test-auc-mean'].max()

#%% Conduct same analysis with "MAE"
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=10
)
cv_results

#%% Print best MAE
cv_results['test-mae-mean'].min()

#%% GridSearch parameter
# You can try wider intervals with a larger step between
# each value and then narrow it down. Here after several
# iterations this an optimal value to start.
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(9,12)
    for min_child_weight in range(5,8)
]

#%% Define initial best params for max_depth and min_child_weight
min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

#%% Edit the parameter values
# We get the best score with a max_depth of 10 and min_child_weight of 5, so let's update
# our params
params['max_depth'] = 10
params['min_child_weight'] = 5

#%% Create gridsearch parameters for subsample and colsample
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(7,11)]
    for colsample in [i/10. for i in range(7,11)]
]

#%% Look for best combination of subsample and colsample_bytree
min_mae = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

#%% Update the parameter values
params['subsample'] = .9
params['colsample_bytree'] = .8

#%% Look for best eta
min_mae = float("Inf")
best_params = None
for eta in [.4, .3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))
    # We update our parameters
    params['eta'] = eta
    # Run and time CV
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['mae'],
            early_stopping_rounds=10
          )
    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta
print("Best params: {}, MAE: {}".format(best_params, min_mae))

#%% Update the eta value
params['eta'] = .05

#%% Create final parameters
params = {'max_depth': 10,
 'min_child_weight': 5,
 'eta': 0.05,
 'subsample': .8,
 'colsample_bytree': 0.9,
 'objective': 'binary:hinge',
 'eval_metric': 'error'}

#%% Train final model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)

print("Best error: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))

#%%
accuracy_score(model.predict(dtest), y_test)

#%%
model.save_model("xgboost_target2.model")

#%% Quickly attempting the XGBoostClassification
xgbc = xgb.XGBClassifier(max_depth=10, min_child_weight=5, learning_rate=.05,
                  subsample=.8, colsample_bytree=.9, n_jobs=-1, random_state=67)

#%% Fit and predict
model = xgbc.fit(X_train, y_train)
y_pred = model.predict(X_test)

#%% Score
print(accuracy_score(y_pred, y_test))
