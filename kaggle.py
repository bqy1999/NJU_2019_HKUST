import pandas as pd
from pandas import DataFrame
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split

x_train = pd.read_csv('train.csv', header=0, sep = ',')
x_test = pd.read_csv('test.csv', header=0, sep = ',')
y_train_std = x_train['time']
x_train = x_train.drop('time', axis=1)
#  print(x_train.head())
#  print(x_test.head())

x_train = x_train.drop('id', axis=1)
x_test = x_test.drop('id', axis=1)

type_train = pd.get_dummies(x_train['penalty'])
type_test = pd.get_dummies(x_test['penalty'])
x_train = x_train.drop('penalty', axis=1)
x_test = x_test.drop('penalty', axis=1)

alpha_train = np.log10(x_train['alpha'])
alpha_test = np.log10(x_test['alpha'])
x_train = x_train.drop('alpha', axis=1)
x_test = x_test.drop('alpha', axis=1)

#  scaler = preprocessing.StandardScaler().fit(x_train['max_iter'])
#  StandardScaler(copy=True, with_mean=True, with_std=True)
#  x_train['max_iter'] = scaler.transform(x_train['max_iter'])
#  x_test['max_iter'] = scaler.transform(x_test['max_iter'])
#  scaler = preprocessing.StandardScaler().fit(x_train['random_state'])
#  StandardScaler(copy=True, with_mean=True, with_std=True)
#  x_train['random_state'] = scaler.transform(x_train['random_state'])
#  x_test['random_state'] = scaler.transform(x_test['random_state'])

scaler = preprocessing.StandardScaler().fit(x_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
x_train = pd.DataFrame(scaler.transform(x_train))
x_test = pd.DataFrame(scaler.transform(x_test))

x_train = pd.concat([x_train, type_train], axis=1)
x_test = pd.concat([x_test, type_test], axis=1)
x_train_std = pd.concat([x_train, alpha_train], axis=1)
x_test_std = pd.concat([x_test, alpha_test], axis=1)

#  print(type_train.head())
#  print(type_test.head())

print(x_train_std.head())
print(x_test_std.head())

#  model = SVR(kernel='rbf')
#  param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 10000], 'gamma': [0.01, 0.001, 0.0001]}
#  grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)
#  grid_search.fit(x_train_std, y_train_std)
#  best_parameters = grid_search.best_estimator_.get_params()
#  for para, val in list(best_parameters.items()):
    #  print(para, val)
#  model = SVR(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])
#  model.fit(x_train_std, y_train_std)
#  print(model.score(x_train_std, y_train_std))

#  model = SGDRegressor(loss="squared_loss", learning_rate="invscaling")
#  param_grid = {'max_iter': [5, 50, 500, 5000, 50000, 500000]}
#  grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)
#  grid_search.fit(x_train_std, y_train_std)
#  best_parameters = grid_search.best_estimator_.get_params()
#  for para, val in list(best_parameters.items()):
    #  print(para, val)
#  model = SGDRegressor(loss="squared_loss", learning_rate="invscaling", max_iter = best_parameters['max_iter'])
#  model.fit(x_train_std, y_train_std)
#  print(model.score(x_train_std, y_train_std))

X_train,X_test,y_train,y_test = train_test_split(x_train_std, y_train_std, test_size=0.2)

quadratic_featurizer = PolynomialFeatures(degree=3)
x_train_std = quadratic_featurizer.fit_transform(x_train_std)
x_test_std = quadratic_featurizer.transform(x_test_std)
print(x_train_std)

#  model = LinearRegression(fit_intercept=True, copy_X=True, normalize=True)
#  param_grid = {'n_jobs': [10, 15, 100, 1000, 10000, 100000]}
#  grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)
#  grid_search.fit(X_train, y_train)
#  best_parameters = grid_search.best_estimator_.get_params()
#  for para, val in list(best_parameters.items()):
    #  print(para, val)
#
#  model = LinearRegression(fit_intercept=True, copy_X=True, normalize=True, n_jobs=best_parameters['n_jobs'])
#  model.fit(X_train, y_train)
#  print(mean_squared_error(model.predict(X_test), y_test))

model = ElasticNet(fit_intercept=True, normalize=True)
param_grid = {'alpha':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 'max_iter': [10000, 100000, 1000000], 'l1_ratio': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 'tol': [0.0001, 0.001, 0.01, 0.1, 1]}
grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)
grid_search.fit(x_train_std, y_train_std)
best_parameters = grid_search.best_estimator_.get_params()
for para, val in list(best_parameters.items()):
    print(para, val)
model = ElasticNet(alpha=best_parameters['alpha'], max_iter = best_parameters['max_iter'], l1_ratio=best_parameters['l1_ratio'], tol=best_parameters['tol'], fit_intercept=True, normalize=True)
model.fit(x_train_std, y_train_std)
print(model.score(x_train_std, y_train_std))

#  model = Lasso(normalize=True)
#  param_grid = {'alpha':[1, 0.1, 0.001, 0.0005], 'max_iter': [1000, 10000, 100000], 'tol': [0.0001, 0.001, 0.01, 0.1]}
#  grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)
#  grid_search.fit(x_train_std, y_train_std)
#  best_parameters = grid_search.best_estimator_.get_params()
#  for para, val in list(best_parameters.items()):
    #  print(para, val)
#  model = Lasso(normalize=True, alpha=best_parameters['alpha'], max_iter = best_parameters['max_iter'], tol=best_parameters['tol'])
#  model.fit(x_train_std, y_train_std)
#  print(model.score(x_train_std, y_train_std))

model.fit(x_train_std, y_train_std)
#  print(model.score(x_train_std, y_train_std))
ans = pd.DataFrame(columns=['time'], data = model.predict(x_test_std))
y_test_std = ans
y_test_std.to_csv('submit11.csv', sep=',')
