from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np
import preprocessor as pre
import itertools


def set_model(model_type):
    if model_type == "Ridge":
        return Ridge(alpha=100)
    return Lasso(alpha=10)


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

# This method finds the subset of features that leads to the lowest MAE
def subset_selection(feature_names, train_features, train_labels, test_features, test_labels, model) :
    """

    :param feature_names: names of all the features to make subsets out of
    :param train_features: the training features data
    :param train_labels: the training label data
    :param test_features: the testing features data
    :param test_labels: the testing label data
    :param model: the model to predict with
    :return: The best subset
    """

    lowest_mae = 10000000;  # init 'lowest MAE' to a number you know is higher then possible MAEs

    features = []
    for h in range(1, len(feature_names)+1):

        for subset in itertools.combinations(feature_names, h):

            features = train_features

            for i in train_features.columns.values:

                if i not in subset:
                    features = features.drop([i], 1)

            model.fit(features, train_labels) # fit on subset features and train_labels

            price_pred = model.predict(features) # predict on the subset of features

            mae = np.mean(abs(train_labels - price_pred)) # find training error

            if lowest_mae > mae:
                lowest_mae = mae;
                best_features = features.columns.values

    print("\nThe lowest MAE is ", lowest_mae, " made from the features\n", best_features)

if __name__ == '__main__':

    model = set_model("Ridge")

    from load_csv import DataSet

    data = DataSet()
    data = pre.one_hot_encode(data.training)
    train_x = np.array(data.drop(['ClaimAmount', 'rowIndex'], 1))
    train_y = np.array(data['ClaimAmount'])

    coeffs = k_fold_get_coefs(train_x, train_y, K=10, features=data.drop(['ClaimAmount', 'rowIndex'], 1).columns.values, model=model)

    pure_data = DataSet()

    just_features, throwaway = zip(*coeffs)

    set = pre.one_hot_encode(pure_data.training)
    set_x = set.drop(['ClaimAmount', 'rowIndex'], 1)
    set_y = set['ClaimAmount']

    # split it into 25, 75
    test_x = set_x[0:17500]
    test_y = set_y[0:17500]
    train_x = set_x[17501:70000]
    train_y = set_y[17501:70000]

    # can only use best_subset_selection on < 41 features
    subset_selection(just_features[56:64], train_x, train_y, test_x, test_y, model)
