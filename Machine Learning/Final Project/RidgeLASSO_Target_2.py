"""
Ridge LASSO Elastic Project for School

Script below is specifically for target2 using a Ridge LASSO Elastic technique.

@author: Thomas Gow
"""
#%%
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV

#%% Data
train = pd.read_csv(r"/Users/thomasgow/Documents/IAA/Machine Learning/Competition/MLProject_train.csv", engine='python')
valid = pd.read_csv(r"/Users/thomasgow/Documents/IAA/Machine Learning/Competition/MLProject_valid.csv")

#%% Dropping the second target
# Didn't have as much success combining targets into one column
# Decided to train and fit each target one at a time
train = train.drop(columns=['target2'])
valid = valid.drop(columns=['target2'])

#%%
# Dropped the few na columns
train.dropna(inplace=True)
valid.dropna(inplace=True)

# Create a sample for easier training
# Statistically it should be more than enough to get
# statistically sound results
sample = train.sample(n = 10000)

# Create all target and feature data
X_sample = sample.drop(columns=['target1'])
X_train = train.drop(columns=['target1'])
X_test = valid.drop(columns=['target1'])
y_sample = sample['target1']
y_train = train['target1']
y_test = valid['target1']

#%% Logistic Regression
lr = LogisticRegression()
lr.fit(X_sample, y_sample)
accuracy_score(y_test, lr.predict(X_test))
# 0.7468666666666667

#%%
"""
Ridge Regression : In ridge regression, the cost function is altered by adding a penalty 
equivalent to square of the magnitude of the coefficients.
So ridge regression puts constraint on the coefficients (w). The penalty term (lambda) 
regularizes the coefficients such that if the coefficients take large values the optimization 
function is penalized. So, ridge regression shrinks the coefficients and it helps to reduce 
the model complexity and multi-collinearity. Going back to eq. 1.3 one can see that 
when λ → 0 , the cost function becomes similar to the linear regression cost function (eq. 1.2). 
So lower the constraint (low λ) on the features, the model will resemble linear regression model.
"""

#%% Ridge CV analyzing different alpha levels
randr = RidgeCV(alphas=[1, .5, .25, .01, .05, .001, .0005], scoring='r2', normalize=True).fit(X_sample, y_sample)
randr.alpha_

#%% Ridge CV analyzing different alpha levels based on results form prior cell
randr = RidgeCV(alphas=[.3, .29, .28, .27, .26, .25, .24, .23, .22, .21], scoring='r2', normalize=True).fit(X_sample, y_sample)
randr.alpha_

#%% Create final RidgeClassifier
# higher the alpha value, more restriction on the coefficients; low alpha > more generalization,
# coefficients are barely restricted and in this case linear and ridge regression resembles
rc = RidgeClassifier(alpha=.27, normalize=True).fit(X_sample, y_sample)

# predict
y_pred = rc.predict(X_test)

# print accuracy
accuracy_score(y_test, y_pred)
# 0.7754

#%% Printing other statistics
print(pd.Series(rc.coef_[0], index = X_test.columns))

# MSE
print(mean_squared_error(y_test, y_pred))
# 0.2246

# Accuracy
print(accuracy_score(y_test, y_pred))
# 0.7754

# AUC
print(roc_auc_score(y_test, y_pred))
# 0.6381309794896967


#%% LASSO
"""
Lasso Regression : The cost function for Lasso (least absolute shrinkage and selection operator) 
regression can be written as
Just like Ridge regression cost function, for lambda =0, the equation above reduces the equation.
The only difference is instead of taking the square of the coefficients, magnitudes are 
taken into account. This type of regularization (L1) can lead to zero coefficients i.e. some 
of the features are completely neglected for the evaluation of output. So Lasso regression not 
only helps in reducing over-fitting but it can help us in feature selection. 
Just like Ridge regression the regularization parameter (lambda) can be controlled.
"""
#%% LASSO CV test different alphas
lass_cv = LassoCV(alphas=[1, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001, .0005, .0001, .00001], cv=3, tol=.001, random_state=67, normalize=True)

# fit to data
lass_cv.fit(X_sample, y_sample)

# score based off of test data
lass_cv.score(X_sample, y_sample)

# best alpha
lass_cv.alpha_

#%% Test more alphas based off of last cell
lass_cv = LassoCV(alphas=[.008, .009, .0001, .0002, .0003, .0004], cv=3, random_state=67, normalize=True)

# fit to data
lass_cv.fit(X_sample, y_sample)

# score based off of test data
lass_cv.score(X_sample, y_sample)

# best alpha
lass_cv.alpha_

#%% Fit final LASSO
lasso = Lasso(alpha=0.0001, normalize=True)

# fit to data
lasso.fit(X_sample, y_sample)

# predict true test y's
y_pred = lasso.predict(X_test)

# edit predictions to be 1 or 0 based on .5 threshold
y_pred[y_pred >= .5] = 1
y_pred[y_pred < .5] = 0

# score these predictions
test_score = accuracy_score(y_test, y_pred)

# number of coefficients used
coeff_used = np.sum(lasso.coef_!=0)
print("test score: ", test_score)
print("number of features used: ", coeff_used)

#%%
'''
In addition to setting and choosing a lambda value elastic net also allows us to tune 
the alpha parameter where 𝞪 = 0 corresponds to ridge and 𝞪 = 1 to lasso. Simply put, if 
you plug in 0 for alpha, the penalty function reduces to the L1 (ridge) term and if we 
set alpha to 1 we get the L2 (lasso) term. Therefore we can choose an alpha value between 
0 and 1 to optimize the elastic net. Effectively this will shrink some coefficients and 
set some to 0 for sparse selection.
'''

#%% Elastic Net
# testing alphas
elast = ElasticNetCV(alphas=[1, .75, .5, .25, .1, .01, .001, .0001, .00001], tol=.001, normalize=True)

# fitting to data
elast.fit(X_sample, y_sample)

# scoring the data
elast.score(X_sample, y_sample)

# best alpha
elast.alpha_

#%% Testing alphas based off of last cv
elast = ElasticNetCV(alphas=[.008, .009, .0001, .0002, .0003], tol=.001, normalize=True)

# fit to the data
elast.fit(X_sample, y_sample)

# scored the data
elast.score(X_sample, y_sample)

# best alpha
elast.alpha_

#%% Final ElasticNet model
elast = ElasticNet(alpha = 0.0002, normalize=True)

# fit the data
elast.fit(X_train,y_train)

# predict y values
y_pred = elast.predict(X_test)

#%% edit predictions to be 1 or 0 based on .5 threshold
y_pred[y_pred >= .5] = 1
y_pred[y_pred < .5] = 0

#%% Print final metrics
# MSE
print(mean_squared_error(y_test, y_pred))
# 0.5135659646043534

# Accuracy
print(accuracy_score(y_test, y_pred))
# 0.73625
