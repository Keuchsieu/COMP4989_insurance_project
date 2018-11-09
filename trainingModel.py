from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np

MODEL_LIST = [
    "Linear",
    "Polynomial",
    "Ridge",
    "Lasso",
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


def set_model(model_type):
    if model_type == 'Linear':
        return LinearRegression()
    elif model_type == "Ridge":
        return Ridge()
    return Lasso()


def k_fold(x, y, K, func=None, model=None, **kwargs):
    """
    :param x: Feature matrix, type of ndarray
    :param y: Label Vector, type of ndarray
    :param K: number of K fold
    :param func: the function that runs the validation training and predicting
    :param model: a model from sklearn that fit() and predict()
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
        if func:
            y_hat = func(x_cv, Xtrain, Ytrain, **kwargs)
        else:
            model.fit(Xtrain, Ytrain)
            y_hat = model.predict(x_cv)
        error = calculate_error(y_cv, y_hat)
        if min_error == -1 or error < min_error:
            min_error = error
            best_k = i
    return best_k, min_error


if __name__ == '__main__':
    model = set_model("Linear")

    from load_csv import DataSet
    data = DataSet()
    bk, me = k_fold(data.get_testX(), data.get_trainY(), K=10, model=model)
    print(bk, me)
