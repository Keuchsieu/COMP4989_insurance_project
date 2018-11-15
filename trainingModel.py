from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

MODEL_LIST = [
    "Linear",
    "Polynomial",
    "Ridge",
    "Lasso",
]

SUPPLEMENT_MODELS = [
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


def set_model(model_type):
    if model_type == 'Linear':
        return LinearRegression()
    elif model_type == "Ridge":
        return Ridge()
    elif model_type == "kNN":
        pass  # the function is not a model, it returns prediction
        # return modified_knn()
    return Lasso()


def modified_knn(x_cv, Xtrain, Ytrain, **kwargs):
    """
    :param x_cv:
    :param Xtrain:
    :param Ytrain:
    :param kwargs:
    :return:
    """

    for i in range(len(Ytrain)):
        if Ytrain[i] > 0:
            Ytrain[i] = 1
        else:
            Ytrain[i] = int(Ytrain[i])
    knc = KNeighborsClassifier(n_neighbors=1)
    knc.fit(Xtrain, Ytrain)
    return knc.predict(x_cv)


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
        if func:  # if predicting using a written function
            y_hat = func(x_cv, Xtrain, Ytrain, **kwargs)
        else:  # if training a model from sklearn
            model.fit(Xtrain, Ytrain)
            y_hat = model.predict(x_cv)
        error = calculate_error(y_cv, y_hat, **kwargs)
        errors.append(error)
    average_error = np.mean(errors)
    return average_error


if __name__ == '__main__':
    from load_csv import DataSet
    data = DataSet()
    # x_train = data.get_trainX()
    x_train = data.get_noClaimX()
    # y_train = data.get_trainY()
    y_train = data.get_noClaimY()
    errors_cred = []
    for model_name in MODEL_LIST:
        print('Trying model: '+model_name)
        model = set_model(model_name)
        model_error = k_fold(x_train, y_train, 10, model=model)
        errors_cred.append({
            'model_mae': model_error,
            'model_name': model_name
        })
        print('Model mae {} for model {}'.format(model_error, model_name))
