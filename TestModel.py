from load_csv import DataSet
from preprocessor import one_hot_encode
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import tree
import numpy as np
import pandas as pd


class TestModel:

    def __init__(self, ohe=(0, 0), features='all', class_feature='all',
                 classify=True, classifier='svc', c_var=1.0, model='Ridge',
                 m_alpha=1000000, poly_p=1, k_fold=10):
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
        self.model_name = "{}_{}_{}_{}cvar_{}lambda_{}p_{}fold_clsfe{}".format(
            model, ('cls' if classify else 'ncls'), classifier,
            c_var, m_alpha, poly_p,  k_fold, class_feature)
        self.classify = classify
        self.prediction = -1
        self.k_fold = k_fold
        self.data = DataSet()
        self.y_train_os = self.data.get_osY()   # get over-sampled data
        self.y_train_mae = self.data.get_trainY()   # get normal data
        # modify features used in model, pre-processing
        if ohe != (0, 0):
            self.x_train_all = one_hot_encode(self.data.get_trainX_pd(), lower_limit=ohe[0], upper_limit=ohe[1])
            self.x_test_all = one_hot_encode(self.data.get_testX_pd())
            self.model_name += "_L{}U{}".format(ohe[0], ohe[1])
        else:
            self.x_train_all_mae = self.data.get_trainX_pd()    # get normal data
            self.x_train_all_os = self.data.get_osX()   # get over-sampled data
            self.x_test_all = self.data.get_testX_pd()
            self.model_name += "_NON"
        if features == 'all':
            self.x_train = np.array(self.x_train_all_mae)
            self.x_test = np.array(self.x_test_all)
            self.model_name += "_allFeature"
        else:
            self.x_train = np.array(self.x_train_all_mae.loc[:, features])
            self.x_test = np.array(self.x_test_all.loc[:, features])
            for name in features:
                self.model_name += "_" + name
        # classify with different feature set
        # convert np.array (returned from oversampling method) to Pandas DF
        self.x_train_all_os = pd.DataFrame(data=self.x_train_all_os,
                                           columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
                                                    'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
                                                    'feature11', 'feature12', 'feature13', 'feature14', 'feature15',
                                                    'feature16', 'feature17', 'feature18'])
        if class_feature == 'all':
            self.x_class = np.array(self.x_train_all_os)
        else:
            self.x_class = np.array(self.x_train_all_os.loc[:, class_feature])

        # check test set size
        if features != 'all':
            assert self.x_train.shape[1] == self.x_test.shape[1], \
                "Number of features doesn't match between test set({}) and training set({})".format(self.x_train.shape[1], self.x_test.shape[1])
        # Regression Model setup
        if model == 'Ridge':
            self.model = Ridge(alpha=m_alpha)
        elif model == 'Lasso':
            self.model = Lasso(alpha=m_alpha)
        elif model == 'NN': # if you wish to use NN, use standardized data (I didn't re-add in here b/c NN took long time and got worse
            self.model = MLPRegressor(hidden_layer_sizes=(12, 12, 12), alpha=c_var, max_iter=2000)     # 3 layers, one neuron per feature (7)
        elif model == 'tree':
            self.model = tree.DecisionTreeRegressor(criterion='mae')
        else:
            self.model = LinearRegression()
        if poly_p != 1:  # polynomial feature if wanted
            self.model = make_pipeline(PolynomialFeatures(poly_p), self.model)
        # Classification Model setup
        if classifier == 'knn':
            self.classifier = KNeighborsClassifier(n_neighbors=c_var)
        elif classifier == 'svc':
            self.classifier = SVC(C=c_var, kernel='rbf')
        elif classifier == 'gnb':
            self.classifier = GaussianNB()
        elif classifier == 'mnb':
            self.classifier = MultinomialNB()
        elif classifier == 'bnb':
            self.classifier = BernoulliNB()
        elif classifier == 'lr':
            self.classifier = LogisticRegression(C=c_var)
        elif classifier == 'tree':
            self.classifier = tree.DecisionTreeClassifier()
        elif classifier == 'rfc':
            self.classifier = RandomForestClassifier(n_estimators=115, criterion='entropy', random_state=1)
        elif classifier == 'NN':    # if you wish to use NN, use standardized data (I didn't re-add in here b/c NN took long time and got worse results
            self.classifier = MLPClassifier(hidden_layer_sizes=(7, 7, 7))   # 3 layers with 7 neurons each (one for each feature)

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
        self.model.fit(self.x_train_all_mae, self.y_train_mae)  # use normal data
        self.classifier.fit(self.x_train_all_os, self.y_train_os)   # use over-sampled data
        prediction = self.classifier.predict(self.x_test) * self.model.predict(self.x_test)
        assert max(prediction) != 0
        return prediction

    def get_mae(self, debug=False):
        data_length = self.x_train.shape[0]
        chunk = int(data_length / self.k_fold)
        errors = []
        scores = []
        for i in range(self.k_fold):
            x_cv = []  # features of cv on regression
            y_cv = []
            x_cv_class = []  # feature of cv on classification
            x_train = []  # features selected for regression
            y_train = []
            x_class = []  # features selected for classification
            # separate cv set and training set
            for j in range(data_length):
                if debug:
                    # print("j: {}".format(j))
                    pass
                if int(j / chunk) == i:
                    # concatenate entire row
                    x_cv.append(self.x_train[j, :])
                    # concatenate one value
                    y_cv.append(self.y_train_mae[j])
                    x_cv_class.append(self.x_class[j, :])
                else:
                    x_train.append(self.x_train[j, :])
                    y_train.append(self.y_train_mae[j])
                    x_class.append(self.x_class[j, :])
            x_cv = np.array(x_cv)
            y_cv = np.array(y_cv)
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_class = np.array(x_class)
            if debug:
                print('Iteration {}'
                      '\nShape of x_train: {}'
                      '\nShape of y_train: {}'
                      '\nShape of x_class: {}'.format(i, x_train.shape, y_train.shape, x_class.shape))
            self.model.fit(x_train, y_train)
            y_hat = self.model.predict(x_cv)
            if self.classify:
                y_class_train = []
                for val in y_train:
                    y_class_train.append(0 if val == 0 else 1)
                self.classifier.fit(x_class, y_class_train)
                y_class_cv = []
                for val in y_cv:
                    y_class_cv.append(0 if val == 0 else 1)
                y_class_pred = self.classifier.predict(x_cv_class)
                true_pos = 0
                false_pos = 0
                false_neg = 0
                for m in range(0, len(y_class_pred)):
                    true_pos += 1 if (y_class_pred[m] == y_class_cv[m] == 1) else 0
                    false_pos += 1 if (y_class_pred[m] == 1 and y_class_cv[m] == 0) else 0
                    false_neg += 1 if (y_class_pred[m] == 0 and y_class_cv[m] == 1) else 0
                f1 = 0
                if true_pos != 0:
                    f1_p = true_pos / (true_pos + false_pos)
                    f1_r = true_pos / (true_pos + false_neg)
                    f1 = 2 * ((f1_p * f1_r) / (f1_p + f1_r))
                scores.append(f1)
                y_hat *= y_class_pred
            mae = np.mean(np.abs(np.subtract(y_hat, y_cv)))
            errors.append(mae)
        if debug:
            print("Size of error: {} should match number of k fold: {}".format(len(errors), self.k_fold))
        kfold_mae = np.mean(errors)
        kfold_f1 = np.mean(scores)
        self.prediction = kfold_mae
        return kfold_mae, kfold_f1

    def get_f1_only(self, debug=False):
        data_length = self.x_class.shape[0]
        chunk = int(data_length / self.k_fold)
        scores = []
        for i in range(self.k_fold):
            x_cv_class = []
            y_cv = []
            y_train = []
            x_class = []
            for j in range(data_length):
                if int(j / chunk) == i:
                    x_cv_class.append(self.x_class[j, :])
                    y_cv.append(self.y_train_os[j])
                else:
                    x_class.append(self.x_class[j, :])
                    y_train.append(self.y_train_os[j])
            x_cv_class = np.array(x_cv_class)
            y_cv = np.array(y_cv)
            x_class = np.array(x_class)
            y_train = np.array(y_train)
            if debug:
                print('Iteration {}'
                      '\nShape of x_train: {}'
                      '\nshape of y train: {}'
                      '\nshape of x cv:  {}'.format(i, x_class.shape, y_train.shape, x_cv_class.shape))
            y_class_train = []
            for val in y_train:
                y_class_train.append(0 if val == 0 else 1)
            self.classifier.fit(x_class, y_class_train)
            y_class_cv = []
            for val in y_cv:
                y_class_cv.append(0 if val == 0 else 1)
            y_class_pred = self.classifier.predict(x_cv_class)
            true_pos = 0
            false_pos = 0
            false_neg = 0
            if debug:
                print("size of predition {}, size of cv {}".format(len(y_class_pred), len(y_class_cv)))
            for m in range(0, len(y_class_pred)):
                true_pos += 1 if (y_class_pred[m] == y_class_cv[m] == 1) else 0
                false_pos += 1 if (y_class_pred[m] == 1 and y_class_cv[m] == 0) else 0
                false_neg += 1 if (y_class_pred[m] == 0 and y_class_cv[m] == 1) else 0
            if debug:
                print("True pos={}, False pos={}, False neg={}".format(true_pos, false_pos, false_neg))
            f1 = 0
            if true_pos != 0:
                f1_p = true_pos / (true_pos + false_pos)
                f1_r = true_pos / (true_pos + false_neg)
                f1 = 2 * ((f1_p * f1_r) / (f1_p + f1_r))
            scores.append(f1)
        kfold_f1 = np.mean(scores)
        if debug:
            print("F1 score is {}".format(kfold_f1))
        return kfold_f1


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

    # Current best model:
    #    Classification: Use ALL features but feature9, feature6, feature17, feature15, feature12, feature11, feature7
    #       with Random Forest and n_estimators = 115
    #    Prediction: Use features 'feature1', 'feature2', 'feature3', 'feature14', 'feature15', 'feature16'
    #       with Decision Tree Regressor and criterion = 'mae'

    x = TestModel(features=('feature1', 'feature2', 'feature3','feature4', 'feature6','feature7', 'feature13', 'feature17','feature18','feature14', 'feature15', 'feature16'),
                  class_feature=('feature1', 'feature2', 'feature3','feature4','feature5','feature8','feature10','feature13', 'feature14','feature16', 'feature18'),
                  classify=True, classifier='rfc', c_var=4, model="Ridge", m_alpha=100000000000, k_fold=10)
    #f1ss = x.get_f1_only()
    #print("F1: ", f1ss)
    mae, f1ss = x.get_mae()
    print("MAE: ", mae)
    # error, score = x.get_mae()
    # pred_test = x.predict_test()
    # print("{} with MAE: {}".format(x, error))
    # print("{} with F1: {}".format(x, score))
    #
    # from FileWriter import FileWriter
    # print(pred_test.shape)
    # w = FileWriter(file_name=x, data=pred_test)
    # w.write()
