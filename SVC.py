import pandas as pd
import numpy as np
from sklearn.svm import SVC
from load_csv import DataSet
import preprocessor

# Perform K-Folds Cross Validation on the variable Lambda.
# Uses Lasso Shrinkage/Regression to train and test the data.
# Inputs:
# x: training input
# y: training output
# p: lambda value
# K: number of folds
# Returns: cv_error (the average MAE cross validation error), train_error (the average MAE training error)
def kfoldCV_svc(x, y, p, K):
    mae_array = [0] * K
    mae1_array = [0] * K
    d_x = [0] * 5
    d_y = [0] * 5
    for_range = [0] * 6

    num_rows = x.shape[0]
    num_rows = int(num_rows / K)

    svc = SVC(kernel='linear', C=p)  # create SVC

    for j in range(0, K):
        ind_1 = j * num_rows
        ind_2 = (j + 1) * num_rows

        train_1x = x[:ind_1]  # first half of training features (total is 4 data sets)
        valid_1x = x[ind_1:ind_2]  # validation features (1 set of data)
        train_2x = x[ind_2:]  # second half of training features (total is 4 data sets)

        train_1y = y[:ind_1]  # first half of training labels (total is 4 data sets)
        valid_1y = y[ind_1:ind_2]  # validation labels (1 set of data)
        train_2y = y[ind_2:]  # second half of training labels (total is 4 data sets)

        train_set_features = np.concatenate([train_1x, train_2x])  # put the halves together
        train_set_labels = np.concatenate([train_1y, train_2y])  # put the halves together
        cv_features = valid_1x
        cv_labels = valid_1y

        svc.fit(train_set_features, train_set_labels)  # find the coefficients of the model with Lasso
        val = svc.predict(cv_features)  # find for cross validation error

        mean = np.mean(abs(val - cv_labels))  # find the mean for cross validation error

        mae_array[j] = mean

    # print(mae_array[j])

    cv_error = np.mean(mae_array)  # find the mean average for cross validation error

    return (cv_error)


dataset = DataSet()

data = dataset.get_training()

train_ratio = 0.75
# number of samples in the data_subset
num_rows = data.shape[0]

# calculate the number of rows for training
train_set_size = int(num_rows * train_ratio)
# training set: take the first 'train_set_size' rows
train_set = data[:train_set_size]
# test set: take the remaining rows
test_set = data[train_set_size:]

train_x = train_set.drop(['ClaimAmount', 'rowIndex'], 1)

train_y = train_set['ClaimAmount']

num = train_y.shape[0]

print(train_y)

for i in range(num):
    if(train_y[i] == 0.0):
        train_y[i] = int(0)
    else:
        train_y[i] = int(1)

print(train_y)

C = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 100]

c_error = 0
cv_errors = [0] * 1000
best_k = 0
lowest_cv_error = 1000000000

for i in range(len(C)):
    c_error = kfoldCV_svc(train_x, train_y, C[i], 10)
    cv_errors[i] = c_error
    if c_error < lowest_cv_error:
        best_k = C[i]  # set best_k value
        lowest_cv_error = c_error  # set lowest cv error

print(best_k)