"""
This script is combining all models and voting per each observation.

Script below is specifically for target1.

@author: Thomas Gow
"""
#%% Library
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Lasso
import xgboost as xgb

#%% Data
# The training and validation data were given for project
train = pd.read_csv(r"/Users/thomasgow/Documents/IAA/Machine Learning/Competition/MLProject_train.csv", engine='python')
valid = pd.read_csv(r"/Users/thomasgow/Documents/IAA/Machine Learning/Competition/MLProject_valid.csv")

#%% Dropping the second target
# Didn't have as much success combining targets into one column
# Decided to train and fit each target one at a time
# Drop target2 column
train = train.drop(columns=['target2'])
valid = valid.drop(columns=['target2'])

#%% Create clean data and appropriate dataframes
train.dropna(inplace=True)
valid.dropna(inplace=True)

# Create sample for easier tuning
sample = train.sample(n = 10000)

# Create appropriate dataframes
X_sample = sample.drop(columns=['target1'])
X_train = train.drop(columns=['target1'])
X_test = valid.drop(columns=['target1'])
y_sample = sample['target1']
y_train = train['target1']
y_test = valid['target1']

#%% RandomForestClassifier
rf = RandomForestClassifier(bootstrap=True, max_depth = 160,
                            max_features = 3, min_samples_leaf = 2,
                            min_samples_split = 4, n_estimators = 250)
# Fit it to data
rf.fit(X_train, y_train)

# predict
rf_pred = rf.predict(X_test)

# print accuracy
print(accuracy_score(y_test, rf_pred))

#%% Creating dmatrices that are easier for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Setting parameters
params = {'max_depth': 9,
 'min_child_weight': 6,
 'eta': 0.05,
 'subsample': 1,
 'colsample_bytree': 0.7,
 'objective': 'binary:hinge',
 'eval_metric': 'error'}

num_boost_round = 999

# Training on parameters
xgb = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)

# Predict
xgb_pred = xgb.predict(dtest)

# Print accuracy
print(accuracy_score(y_test, xgb_pred))

#%% RidgeClassifier
# higher the alpha value, more restriction on the coefficients; low alpha > more generalization,
# coefficients are barely restricted and in this case linear and ridge regression resembles
rc = RidgeClassifier(alpha=.5, normalize=True).fit(X_train, y_train)

# Predict
rc_pred = rc.predict(X_test)

# Print accuracy
print(accuracy_score(y_test, rc_pred))

#%% XGBClassifier
xgbc = xgb.XGBClassifier(max_depth=9, min_child_weight=6, learning_rate=.05,
                  subsample=1, colsample_bytree=.7, n_jobs=-1, random_state=67)

# fit model
model = xgbc.fit(X_train, y_train)

# predict y_test
xgbc_pred = model.predict(X_test)

# print accuracy
print(accuracy_score(xgbc_pred, y_test))

#%% Lasso from final alpha script for target1
lasso = Lasso(alpha=0.0001, normalize=True)

# fit on data
lasso.fit(X_train, y_train)

# predict
y_pred_lasso = lasso.predict(X_test)

# Create threshold for above and below .5
y_pred_lasso[y_pred_lasso >= .5] = 1
y_pred_lasso[y_pred_lasso < .5] = 0

# print accuracy score
print(accuracy_score(y_test, y_pred_lasso))

#%% Edit into correct format
rf_pred = pd.DataFrame(rf_pred)
rf_pred.columns = ["rf_pred"]
xgb_pred = pd.DataFrame(xgb_pred)
xgb_pred.columns = ["xgb_pred"]
rc_pred = pd.DataFrame(rc_pred)
rc_pred.columns = ["rc_pred"]
xgbc_pred = pd.DataFrame(xgbc_pred)
xgbc_pred.columns = ["xgbc_pred"]
y_pred_lasso = pd.DataFrame(y_pred_lasso)
y_pred_lasso.columns = ["y_pred_lasso"]

# merge
final_pred = pd.merge(rf_pred, xgb_pred, how='outer', left_index=True, right_index=True)
final_pred = pd.merge(final_pred, rc_pred, how='outer', left_index=True, right_index=True)
final_pred = pd.merge(final_pred, xgbc_pred, how='outer', left_index=True, right_index=True)
final_pred = pd.merge(final_pred, y_pred_lasso, how='outer', left_index=True, right_index=True)

#%% Create final prediction based off of voting system
final_pred['final'] = (final_pred['rf_pred'] + final_pred['xgb_pred']+final_pred['rc_pred']+final_pred['xgbc_pred']+final_pred['y_pred_lasso'])/5

#%% Create final predictions based off thresholds
final_pred[final_pred['final'] >= .5] = 1
final_pred[final_pred['final'] < .5] = 0
#%% Score accuracy
y_pred = final_pred['final']
print(accuracy_score(y_test, y_pred))
