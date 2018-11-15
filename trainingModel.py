from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import xlsxwriter # write to excel spreadsheet

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
    min_error = -1
    best_k = 0
    errors = []
    for i in range(1,K):
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
        if min_error == -1 or error < min_error:
            min_error = error
            best_k = i
    average_error = np.mean(errors)
    return best_k, min_error, average_error


if __name__ == '__main__':

    model = set_model("Linear") 
    method = 'mae' # change the method (or function) ex. mae or kNN
    modelName = "Linear"

    from load_csv import DataSet
    data = DataSet()

    workbook = xlsxwriter.Workbook('results.xlsx')
    worksheet = workbook.add_worksheet()
    bold = workbook.add_format({'bold': True})
    worksheet.write('A1', 'Model', bold)
    worksheet.write('B1', 'K.F Rnge', bold)
    worksheet.write('C1', 'Best K.F', bold)
    worksheet.write('D1', 'Found', bold)
    worksheet.write('E1', 'Avg. Error', bold)
    worksheet.write('F1', 'Min. Error', bold)

    k_values = [10,20,30] # K-Folds ranges you wish to use
    models = [set_model("Linear"), set_model("Ridge"), set_model("Lasso")] # change the models
    modelsNames = ["Linear", "Ridge", "Lasso"] # for the excel file

    for i in range(len(models)):

        for j in range(len(k_values)):

            bk, me, average = k_fold(data.get_trainX(), data.get_trainY(), K=k_values[j], model=models[i], method=method)
            print("Best K fold group number {}, min error: {}, average error {}".format(bk, me, average))
            worksheet.write('A' + str((i*3) + (j+2)), modelsNames[i])
            worksheet.write('B' + str((i*3) + (j+2)), k_values[i])
            worksheet.write('C' + str((i*3) + (j+2)), bk)
            worksheet.write('D' + str((i*3) + (j+2)), method)
            worksheet.write('E' + str((i*3) + (j+2)), average)
            worksheet.write('F' + str((i*3) + (j+2)), me)

    workbook.close()