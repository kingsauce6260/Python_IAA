"""
Random Forest Project for School

Script below is specifically for target2 using a RandomForest technique.

@author: Thomas Gow
"""
#%% Loading in Libraries
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

#%% Data
# The training and validation data were given for project
train = pd.read_csv(r"/Users/thomasgow/Documents/IAA/Machine Learning/Competition/MLProject_train.csv", engine='python')
valid = pd.read_csv(r"/Users/thomasgow/Documents/IAA/Machine Learning/Competition/MLProject_valid.csv")

#%% Dropping the second target
# Didn't have as much success combining targets into one column
# Decided to train and fit each target one at a time
train = train.drop(columns=['target1'])
valid = valid.drop(columns=['target1'])

#%%
# Dropped the few na columns
train.dropna(inplace=True)
valid.dropna(inplace=True)

# Create a sample for easier training
# Statistically it should be more than enough to get
# statistically sound results
sample = train.sample(n = 10000)

# Create all target and feature data
X_sample = sample.drop(columns=['target2'])
X_train = train.drop(columns=['target2'])
X_test = valid.drop(columns=['target2'])
y_sample = sample['target2']
y_train = train['target2']
y_test = valid['target2']

#%% Utilize the randomized search cv listing out parameters to randomly test
# The random search was done because this project was time sensitive
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 2000, num = 15)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 200, num = 19)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 6, 8, 10]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


#%%
# Use the random grid to search for best hyperparameters
# First create the base model to tune in the random search
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42)
# Fit the random search model
rf_random.fit(X_sample, y_sample)

#%% See the best parameters
rf_random.best_params_

#%%
# Definition for evaluating a model
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print('Model Performance')
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

#%% Create a base model to compare to the best random accuracy later
base_model = RandomForestClassifier(n_estimators=10, random_state=42)
base_model.fit(X_sample, y_sample)
base_accuracy = evaluate(base_model, X_test, y_test)


#%% Assigned the best random rf
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)

# Compare the two rfs
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

#%% Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True, False],
    'max_depth': [160, 170, 180, 190, 200],
    'max_features': ['auto'],
    'min_samples_leaf': [1,2,3],
    'min_samples_split': [4,5,6],
    'n_estimators': [250, 300, 400]
}

#%% Create the random forest and the grid search to be completed
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)

#%% Fit the data to the grid search to the data
grid_search.fit(X_sample, y_sample)

#%% See the best parameters
grid_search.best_params_

# Assign it to the variable best_grid
best_grid = grid_search.best_estimator_

#%% Set parameters
params = {'bootstrap': True,
 'max_depth': 160,
 'max_features': 3,
 'min_samples_leaf': 2,
 'min_samples_split': 4,
 'n_estimators': 250}

#%% Set up Classifier and fit it to the data
rf = RandomForestClassifier(params)
rf.fit(X_sample, y_sample)

#%% Create predictions and score
predictions = rf.predict(X_test)
print('Accuracy of {:0.2f}%.'.format(100 * accuracy_score(y_test, predictions)))
accuracy = evaluate(rf, X_test, y_test)

#%% Print Comparison
print('Improvement of {:0.2f}%.'.format( 100 * (accuracy - base_accuracy) / base_accuracy))

#%% Analyzing the results
# Calculating mse
print(mean_squared_error(y_test, predictions))
# 0.23053333333333334

# Calculating AUC
print(roc_auc_score(y_test, predictions))
# 0.5976292215219172

# Classification Report
target_names = ['0', '1']
print(classification_report(y_test, predictions, target_names=target_names))

# Confusion Matrix
confusion_matrix(y_test, predictions)