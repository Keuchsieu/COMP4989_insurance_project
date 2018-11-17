from preprocessor import *
from trainingModel import *
from load_csv import DataSet
import csv
from datetime import datetime as time
import os

UPPER = 10
LOWER = 3
NEIGHBOR = 3
KFOLD = 10
lambda_p = -4

data = DataSet()
# all training features, all being np array
trainX = {
    'claim': data.get_claimX(),
    'noclaim': data.get_noClaimX(),
    'raw': data.get_trainX(),
    'onehot': np.array(one_hot_encode(data.get_trainX_pd(), upper_limit=UPPER, lower_limit=LOWER)),
    'c_oh': np.array(one_hot_encode(data.get_claimX_pd())),
    'n_oh': np.array(one_hot_encode(data.get_noClaimX_pd()))
}

# all training labels, all being np array
trainY = {
    'claim': data.get_claimY(),
    'noclaim': data.get_noClaimY(),
    'raw': data.get_trainY()
}

testX = {
    'onehot': np.array(one_hot_encode(data.get_testX_pd(), upper_limit=UPPER, lower_limit=LOWER)),
    'raw': data.get_testX()
}

# model = LinearRegression()
model = Lasso(alpha=10**lambda_p)
knc = KNeighborsClassifier(n_neighbors=NEIGHBOR)


def calculate_error(y, yhat, method='mae'):
    if method == 'mae':
        mae = np.mean(np.abs(y - yhat))
        return mae
    elif method == 'knn':
        y_local = y
        for i in range(len(y)):
            if y[i] > 0:
                y_local[i] = 1
            else:
                y_local[i] = int(y[i])
        return np.mean(np.abs(y_local-yhat))
    # return none if the method is not specified above
    return None


def my_train_predic(x_t, y_t, x):
    y_c = []
    for i in y_t:
        if i == 0:
            y_c.append(0)
        else:
            y_c.append(1)
    y_c = np.array(y_c)
    x_t = np.array(x_t)
    y_t = np.array(y_t)
    x = np.array(x)
    # print('shape x {}, shape y {}'.format(x_t.shape, np.array(y_c).shape))
    knc.fit(x_t, y_c)
    ifclaim = knc.predict(x)
    model.fit(x_t, y_t)
    claimvalue = model.predict(x)
    prediction = ifclaim * claimvalue
    return prediction


def k_fold(x, y, K):
    data_length = x.shape[0]
    chunk = int(data_length / K)
    errors = []
    for i in range(K):
        x_cv = []  # one chunk size of x
        y_cv = []
        Xtrain = []
        Ytrain = []
        for j in range(data_length):
            if int(j / chunk) == i:
                x_cv.append(x[j])
                y_cv.append(y[j])
            else:
                Xtrain.append(x[j])
                Ytrain.append(y[j])
        p = my_train_predic(Xtrain, Ytrain, x_cv)
        error = calculate_error(p, y_cv)
        errors.append(error)
    average_error = np.mean(errors)
    return average_error


data_type = ['raw', 'onehot']

for t in data_type:
    error = k_fold(trainX[t], trainY['raw'], KFOLD)
    print("Average error of {} data: {}".format(t, error))
    pred_test = my_train_predic(trainX[t], trainY['raw'], testX[t])

    file_dir = './predictions/'
    print(np.array(pred_test).shape)
    now = time.now()
    result_file_name = 'lasso_p{}_{:.2f}_{}_result_{}{}{}.csv'.format(lambda_p,error, t, now.date(), now.hour, now.minute)
    with open(os.path.join(file_dir, result_file_name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(pred_test)
