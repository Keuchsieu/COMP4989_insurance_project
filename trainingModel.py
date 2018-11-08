from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np

MODEL_LIST = [
    "linear",
    "polynomial",
    "ridge",
    "lasso",
    "kNN",
    "SVM",
    "naive bayes",
    "random forest",
    "decision tree",
    "logistic regression",
    "gradient boosting",
    "ANN --our saviour"
]


def calculate_error(y, yhat, method='mae'):
    if method == 'mae':
        mae = np.mean(np.abs(y - yhat))
        return mae
    return None


def set_model(model):
    if model == 'Linear':
        return LinearRegression()
    elif model == "Ridge":
        return Ridge()
    return Lasso()


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
    model = set_model("Linear")
    from load_csv import DataSet
    data = DataSet()
    model.fit(data.get_trainX_pd(), data.get_trainY_pd())
    y_pred = model.predict(data.get_testX_pd())
    print(y_pred)
