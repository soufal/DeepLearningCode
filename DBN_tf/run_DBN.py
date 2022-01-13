import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from dbn.tensorflow import SupervisedDBNRegression

import pandas as pd

# Loading dataset
data = pd.read_csv('data/Process.csv')

# 划分X Y
X, Y = data.iloc[:,1:20], data['Biomass']
print(X.columns.tolist())

train_split = int(data.shape[0] * 0.7)
X_train, X_test, Y_train, Y_test = X[0:train_split], X[train_split:], Y[0:train_split], Y[train_split:]
# Data scaling
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)

# Training
regressor = SupervisedDBNRegression(hidden_layers_structure=[50],
                                    learning_rate_rbm=0.1,
                                    learning_rate=0.01,
                                    n_epochs_rbm=100,
                                    n_iter_backprop=50,
                                    batch_size=5,
                                    activation_function='relu')
regressor.fit(X_train, Y_train)

# Test
X_test = min_max_scaler.transform(X_test)
Y_pred = regressor.predict(X_test)
RMES = mean_squared_error(Y_test, Y_pred)
MAE = mean_absolute_error(Y_test, Y_pred)
R2 = r2_score(Y_test, Y_pred)

# 预测结果存为文件测试
# pd.DataFrame(Y_pred).to_csv('pred.csv', index=False)
print('Done.\nR-squared: %f\nRMSE: %f\nMAE: %f' % (R2, RMES, MAE))
