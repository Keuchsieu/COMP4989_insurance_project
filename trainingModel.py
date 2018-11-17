from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import itertools

MODEL_LIST = [
    "Linear",
    # "Polynomial",
    # "Ridge",
    # "Lasso",
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


def feature_loop(x_train, y_train, features):
    errors_cred = []
    for model_name in MODEL_LIST:
        print('Trying model: '+model_name)
        model = set_model(model_name)
        model_error = k_fold(x_train, y_train, 10, model=model)
        errors_cred.append({
            'model_mae': model_error,
            'model_name': model_name,
            'features': features
        })
        print('Model mae {} for model {} of feature {}'.format(model_error, model_name, features))
    return errors_cred


if __name__ == '__main__':
    from load_csv import DataSet
    data = DataSet()
    col_names = list(data.get_testX_pd())
    # print(col_names)
    x_train = data.get_trainX()
    # x_train = data.get_noClaimX()
    y_train = data.get_trainY()
    # y_train = data.get_noClaimY()
    big_records = []
    min_error = -1
    best_model = ''
    best_features = ''
    for i in range(1, x_train.shape[1]+1):
        combies = itertools.combinations(col_names, i)
        print('loop iteration {}'.format(i))
        # combies are list of combinations of i elements
        for combi in combies:
            x_selected = []
            for element in combi:
                index = col_names.index(element)  # find the index of this feature
                x_feature = x_train[:, [index]]  # get the column vector
                if x_selected == []:
                    x_selected = x_feature
                else:
                    np.concatenate((x_selected, x_feature), axis=1)
            # after for loop, the selected feature should be in x selected list
            one_record = feature_loop(x_selected, y_train, combi)
            big_records.append(one_record)
    for feature in big_records:
        for one in feature:
            if min_error == -1 or one['model_mae'] < min_error:
                min_error = one['model_mae']
                best_features = one['features']
                best_model = one['model_name']



