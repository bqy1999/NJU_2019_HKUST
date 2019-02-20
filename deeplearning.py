import pandas as pd
from pandas import DataFrame
import numpy as np
import sklearn
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.linear_model import LinearRegression
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

x_train = pd.read_csv('train.csv', header=0, sep = ',')
x_test = pd.read_csv('test.csv', header=0, sep = ',')
y_train_std = x_train['time']
x_train = x_train.drop('time', axis=1)

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

x_train = pd.concat([x_train, type_train], axis=1)
x_test = pd.concat([x_test, type_test], axis=1)
x_train_std = pd.concat([x_train, alpha_train], axis=1)
x_test_std = pd.concat([x_test, alpha_test], axis=1)

scaler = preprocessing.StandardScaler().fit(x_train_std)
StandardScaler(copy=True, with_mean=True, with_std=True)
#  x_train = pd.DataFrame(scaler.transform(x_train))
x_test_std = pd.DataFrame(scaler.transform(x_test_std))

#  print(x_train_std.head())
print(x_test_std.head())

# define base mode
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(64, input_shape = (x_train_std.shape[1], ), init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=5000, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
# use 10-fold cross validation to evaluate this baseline model
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, x_train_std, y_train_std, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
pipeline.fit(x_train_std, y_train_std)
ans = pd.DataFrame(columns=['time'], data = pipeline.predict(x_test_std))
y_test_std = ans
y_test_std.to_csv('submit12.csv', sep=',')

#  def train_with_sgd(model, x_train_std, y_train_std, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    #  """
    #  Train RNN Model with SGD algorithm.
    #
    #  Parameters
    #  ----------
    #  model : The model that will be trained
    #  x_train_std : input x
    #  y_train：期望输出值
    #  learning_rate：学习率
    #  nepoch：迭代次数
    #  evaluate_loss_after：loss值估计间隔，训练程序将每迭代evaluate_loss_after次进行一次loss值估计
    #  """
    #  # We keep track of the losses so we can plot them later
    #  losses = []
    #  num_examples_seen = 0
    #  #循环迭代训练，这个for循环每运行一次，就完成一次对所有数据的迭代
    #  for epoch in range(nepoch):
    #      # Optionally evaluate the loss
    #      if (epoch % evaluate_loss_after == 0):
    #          loss = model.calculate_loss(x_train_std, y_train_std)
    #          losses.append((num_examples_seen, loss))
    #          time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    #          print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
    #          # Adjust the learning rate if loss increases
    #          #如果当前一次loss值大于上一次，则调小学习率
    #          if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
    #              learning_rate = learning_rate * 0.5
    #              print("Setting learning rate to %f" % learning_rate)
    #          sys.stdout.flush()
    #          # ADDED! Saving model oarameters
    #          save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
    #      # For each training example...
    #      #对所有训练数据执行一轮SGD算法迭代
    #      for i in range(len(y_train_std)):
    #          # One SGD step
    #          model.sgd_step(x_train_std[i], y_train_std[i], learning_rate)
            #  num_examples_seen += 1
