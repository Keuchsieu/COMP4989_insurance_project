import numpy as np
import pandas as pd
import math
from load_csv import DataSet
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier


##################### PART 2 ############################

# k-Nearest Neighbours
# Inputs:
#	x: a test sample
#	Xtrain: array of input vectors of the training set
#	YTrain: array of output values of the training set
#	k: number of nearest neighbours
# Outputs:
#	y_pred: predicted value for the test sample
def kNN(x, Xtrain, YTrain, k):
    euclidean_arr = [0] * 4
    indexes = [0] * 4
    selected = [0] * 3
    output = [0] * 3

    for i in range(0, 4):
        euclidean = ((Xtrain[i][0] - x[0]) ** 2) + ((Xtrain[i][1] - x[1]) ** 2)  # get euclidean distance
        euclidean_arr[i] = euclidean
        indexes[i] = i

    list3 = zip(euclidean_arr, indexes)
    list3 = sorted(list3, key=lambda x: x[0])

    for j in range(0, 3):
        output[j] = YTrain[list3[j][1]]

    counts = np.bincount(output)
    return np.argmax(counts)  # return most occurring number


# data = [[1, 1], [2, 3], [3, 2], [3, 4], [2, 5]]
# output = [0, 0, 0, 1, 1]
# sample = 0
# my_knn_errors = 0
# lib_knn_errors = 0
# neigh = KNeighborsClassifier(n_neighbors=20)
#
# for i in range(0, 5):
#
#     t_set = []
#     t_labels = []
#     x = [[1, 1], [2, 3], [3, 2], [3, 4], [2, 5]]
#
#     for j in range(0, 5):
#
#         if i != j:
#             continue
#         t_set = x
#         t_set.remove(t_set[j])  # create training set by removing validation sample
#         t_labels = np.delete(output, j)  # create training label set by removing validation samples label
#         v_set = data[j]  # create validation set
#
#     y_pred = kNN(v_set, t_set, t_labels, 20)
#     if y_pred != output[i]:
#         my_knn_errors = my_knn_errors + 1;  # increment error count for my function
#     neigh.fit(t_set, t_labels)  # train model
#     lib_y_pred = neigh.predict([v_set])
#     if lib_y_pred != output[i]:
#         lib_knn_errors = lib_knn_errors + 1;  # increment error count for library funciton
#
# print()
# print("PART 2 #3. My kNN error rate is: ", my_knn_errors / len(output))
# print("PART 2 #4. Library kNN rate is: ", lib_knn_errors / len(output))
#

##################### PART 3 ############################

# kfold C.V. for k-Nearest-Neighbours
# Inputs:
# 	Xtrain = array of input vectors as the training set
# 	Ytrain = array of output vectors as the training set
#	K: number of folds
# 	k = number of neighbours
# Output:
#	cv_errors: the average mae from c.v
def kfoldCV_knn(Xtrain, Ytrain, K, k):
    cv_errors = [0] * K

    v_size = int(len(Xtrain) / K)

    for i in range(K):
        start = v_size * i  # index to start slice for v_set
        end = v_size * (i + 1)  # index to end slice for v_set
        v_set = Xtrain[start:end]  # create validation set
        v_labels = Ytrain[start:end]  # create validation labels set
        t_set = np.delete(Xtrain, np.s_[start:end], 0)  # create training set by removing validation samples
        t_labels = np.delete(Ytrain, np.s_[start:end], 0)  # create training labels set by removing validation labels

        neigh = KNeighborsClassifier(n_neighbors=k)  # create kNN and set nearest neighbours to k
        neigh.fit(t_set, t_labels)  # train model

        cv_y_pred = neigh.predict(v_set)  # get pred for validation set
        cv_errors[i] = np.mean(cv_y_pred != v_labels)  # find mean validation error

    return np.mean(cv_errors)  # return MAE from cv


data = DataSet()

data_X = data.get_claimX()
data_Y = data.get_claimY()

n = data_X.shape[0]
split_data = int(n * 0.75)

train_x = data_X[:split_data]
train_y = data_Y[:split_data]
test_x = data_X[:split_data]
test_y = data_Y[:split_data]

k_values = range(1, 21, 2)
cv_errors = [0] * len(k_values)
best_k = 0
lowest_cv_error = 10000000000000

for i in range(len(k_values)):
    c_error = kfoldCV_knn(train_x, train_y, 10, k_values[i])
    cv_errors[i] = c_error
    if c_error < lowest_cv_error:
        best_k = k_values[i]  # set best_k value
        lowest_cv_error = c_error  # set lowest cv error

plt.plot(k_values, cv_errors, label='C.V. Error')
plt.title("Cross Validation Error Rate for kNN k values")
plt.xlabel("k")
plt.ylabel("Average Error Rate")
plt.legend()
plt.show()

print()
print("PART 3. #5: Best 'k' value:", best_k, "\nIt's error rate is", lowest_cv_error)

neigh = KNeighborsClassifier(n_neighbors=best_k)  # set kNN to use best k
neigh.fit(train_x, train_y)  # train model
test_y_pred = neigh.predict(test_x)
test_errors = 0

for i in range(len(test_y_pred)):
    if test_y_pred[i] != test_y[i]:
        test_errors = test_errors + 1  # increment test errors counter

test_error_rate = test_errors / len(test_y_pred)

print()
print("PART 3. #6: Test Error Rate:", test_error_rate)


