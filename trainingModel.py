from sklearn import linear_model
import numpy as np


def calculate_error(y, yhat, method='mae'):
    if method == 'mae':
        mae = np.mean(np.abs(y - yhat))
        return mae
    return None


def k_fold(x, y, func, K, **kwargs):
    """
    :param x: Feature matrix, type of ndarray
    :param y: Label Vector, type of ndarray
    :param func: the function that runs the validation training and predicting
    :param K: number of K fold
    :param kwargs: arguments that functions differently for different model
    :return: the error rate of the data
    """
    data_length = x.shape[0]
    chunk = int(data_length / K)
    min_error = -1
    best_k = 0
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
        y_hat = func(x_cv, Xtrain, Ytrain, **kwargs)
        error = calculate_error(y, y_hat)
        if min_error == -1 or error < min_error:
            min_error = error
            best_k = i
    return best_k, min_error


if __name__ == '__main__':
    model = linear_model.LinearRegression()
    from load_csv import DataSet
    data = DataSet()
    model.fit(data.get_trainX_pd(), data.get_trainY_pd())
    y_pred = model.predict(data.get_testX_pd())
    print(y_pred)
