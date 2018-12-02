from load_csv import DataSet
from preprocessor import one_hot_encode
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np


class TestModel:

    def __init__(self, ohe=(0, 0), features='all',
                 classify=True, classifier='knn', c_var=1, model='Linear',
                 m_alpha=1, poly_p=1, k_fold=10):
        """
        Constructor of test model
        :param ohe: Boolean, if want to one hot encode
        :param features: Tuple/list of String,features selected to train
        :param classify: Boolean, if use classify before regression
        :param classifier: String, classification model selected for training
        :param c_var: value used in the classification model
        :param model: String, regression model selected for training
        :param m_alpha: lambda used in Ridge and Lasso models
        :param poly_p: useful only when it is not 1, will create polynomial model based on given value
        :param k_fold: number of k folds to test with
        """
        self.model_name = "{}_{}_{}_{}cvar_{}lambda_{}p_{}fold".format(
            model, ('cls' if classify else 'ncls'), classifier,
            c_var, m_alpha, poly_p,  k_fold)
        self.classify = classify
        self.prediction = -1
        self.k_fold = k_fold
        self.data = DataSet()
        self.y_train = self.data.get_trainY()
        # modify features used in model, pre-processing
        if ohe != (0, 0):
            self.x_train_all = one_hot_encode(self.data.get_trainX_pd(), lower_limit=ohe[0], upper_limit=ohe[1])
            self.x_test_all = one_hot_encode(self.data.get_testX_pd())
            self.model_name += "_L{}U{}".format(ohe[0], ohe[1])
        else:
            self.x_train_all = self.data.get_trainX_pd()
            self.x_test_all = self.data.get_testX_pd()
            self.model_name += "_NON"
        if features == 'all':
            self.x_train = np.array(self.x_train_all)
            self.x_test = np.array(self.x_test_all)
            self.model_name += "_allFeature"
        else:
            self.x_train = np.array(self.x_train_all.loc[:, features])
            self.x_test = np.array(self.x_test_all.loc[:, features])
            for name in features:
                self.model_name += "_" + name

        assert self.x_train.shape[1] == self.x_test.shape[1], \
            "Number of features doesn't match between test set({}) and training set({})".format(self.x_train.shape[1], self.x_test.shape[1])
        # Regression Model setup
        if model == 'Ridge':
            self.model = Ridge(alpha=m_alpha)
        elif model == 'Lasso':
            self.model = Lasso(alpha=m_alpha)
        else:
            self.model = LinearRegression()
        if poly_p != 1:  # polynomial feature if wanted
            self.model = make_pipeline(PolynomialFeatures(poly_p), self.model)
        # Classification Model setup
        if classifier == 'knn':
            self.classifier = KNeighborsClassifier(n_neighbors=c_var)
        if classifier == 'svc':
            self.classifier = SVC(C=c_var, kernel='linear')

    def __str__(self):
        """
        String function
        :return: model name of this modified model
        """
        if self.prediction == -1:
            return self.model_name + '_unpredicted'
        else:
            return self.model_name + '_{:.4f}'.format(self.prediction)

    def __add__(self, other):
        return self.__str__() + str(other)

    def predict_test(self):
        # fit the entire training sets
        self.model.fit(self.x_train, self.y_train)
        if self.classify:
            y_class = []
            for val in self.y_train:
                y_class.append(0 if val == 0 else 1)
            self.classifier.fit(self.x_train, y_class)
            return self.classifier.predict(self.x_test) * self.model.predict(self.x_test)
        else:
            return self.model.predict(self.x_test)

    def get_mae(self, debug=False):
        data_length = self.x_train.shape[0]
        chunk = int(data_length / self.k_fold)
        errors = []
        for i in range(self.k_fold):
            x_cv = []
            y_cv = []
            x_train = []
            y_train = []
            # separate cv set and training set
            for j in range(data_length):
                if debug:
                    print("j: {}".format(j))
                if int(j / chunk) == i:
                    # concatenate entire row
                    x_cv.append(self.x_train[j, :])
                    # concatenate one value
                    y_cv.append(self.y_train[j])
                else:
                    x_train.append(self.x_train[j, :])
                    y_train.append(self.y_train[j])
            x_cv = np.array(x_cv)
            y_cv = np.array(y_cv)
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            if debug:
                print('Iteration {}'
                      '\nShape of x_train: {}'
                      '\nShape of y_train: {}'.format(i, x_train.shape, y_train.shape))
            self.model.fit(x_train, y_train)
            y_hat = self.model.predict(x_cv)
            if self.classify:
                y_class = []
                for val in y_train:
                    y_class.append(0 if val == 0 else 1)
                self.classifier.fit(x_train, y_class)
                y_hat *= self.classifier.predict(x_cv)
            mae = np.mean(np.abs(np.subtract(y_hat, y_cv)))
            errors.append(mae)
        if debug:
            print("Size of error: {} should match number of k fold: {}".format(len(errors), self.k_fold))
        kfold_mae = np.mean(errors)
        self.prediction = kfold_mae
        return kfold_mae


if __name__ == '__main__':
    """
    USAGE of TestModel:
    1. Construct model with values you want, please check constructor to see what options do we have now
    2. Use get_mae() to get K fold mae result, and use this to compare with other models you created
    3. (optional) If the model is good compared with other model, run predict_test() to get the vector of prediction
       on the test set.
    4. (optional) Output the prediction file into ./predictions/ folder, 
       rename it to testsetassessment_group_subnumber.csv and upload to d2l folder.
       AND complete the model_completion google sheet to record it
    """
    x = TestModel(features=('feature14', 'feature17', 'feature8'), classify=True, classifier='knn', c_var=1, k_fold=2) # used 7 features
    #error = x.get_mae(debug=True)
    pred_test = x.predict_test()
    #print("{} with MAE: {}".format(x, error))
    #from FileWriter import FileWriter
    #print(pred_test.shape)
    #w = FileWriter(file_name=x, data=pred_test)
    #w.write()
