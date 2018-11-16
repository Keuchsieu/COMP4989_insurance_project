from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np
import preprocessor as pre

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
    coefficients = 0
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
            coefficients += np.abs(model.coef_)
            y_hat = model.predict(x_cv)
        error = calculate_error(y_cv, y_hat)
        if min_error == -1 or error < min_error:
            min_error = error
            best_k = i
    return best_k, min_error, coefficients


if __name__ == '__main__':
    model = set_model("Lasso")

    from load_csv import DataSet
    data = DataSet()
    data = pre.one_hot_encode(data.training)
    train_x = np.array(data.drop(['ClaimAmount', 'rowIndex'], 1))
    train_y = np.array(data['ClaimAmount'])

    bk, me, coeffs = k_fold(train_x, train_y, K=10, model=model)

    data = data.drop(['ClaimAmount', 'rowIndex'], 1)
    dictionary = dict(zip(data.columns.values, coeffs))
    sorted_d = sorted(dictionary.items(), key=lambda x: x[1])

    print(sorted_d)





