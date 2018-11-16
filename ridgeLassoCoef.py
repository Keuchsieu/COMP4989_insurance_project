from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np
import preprocessor as pre


def set_model(model_type):
    if model_type == "Ridge":
        return Ridge()
    return Lasso()


def k_fold_get_coefs(x, y, K, features, model=None):
    """
    :param x: Feature matrix, type of ndarray
    :param y: Label Vector, type of ndarray
    :param K: number of K fold
    :param model: a model from sklearn that fit() and predict(), either ridge or lasso
    :return: Ordered 2D array of features and their coefficients
    """
    data_length = x.shape[0]
    chunk = int(data_length / K)
    coefficients = 0
    for i in range(K):
        Xtrain = []
        Ytrain = []
        for j in range(data_length):
            if int(j / chunk) != i:
                Xtrain.append(x[j])
                Ytrain.append(y[j])
        model.fit(Xtrain, Ytrain)
        coefficients += np.abs(model.coef_)
        dictionary = dict(zip(features, coefficients))
        sorted_coefs = sorted(dictionary.items(), key=lambda x: x[1])
    return sorted_coefs


if __name__ == '__main__':
    model = set_model("Lasso")

    from load_csv import DataSet
    data = DataSet()
    data = pre.one_hot_encode(data.training)
    train_x = np.array(data.drop(['ClaimAmount', 'rowIndex'], 1))
    train_y = np.array(data['ClaimAmount'])

    coeffs = k_fold_get_coefs(train_x, train_y, K=10,
                              features=data.drop(['ClaimAmount', 'rowIndex'], 1).columns.values, model=model)
    print(coeffs)
