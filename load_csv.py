import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN # SMOTE and ADASYN performed worse
from sklearn.preprocessing import StandardScaler # used to prep data for NN

class DataSet:
    def __init__(self):
        ros = RandomOverSampler(random_state=0)

        self.training = pd.read_csv('./datasets/trainingset.csv')
        testset = pd.read_csv('./datasets/testset.csv')
        self.trainingX = self.training.drop(['rowIndex', 'ClaimAmount'], 1)  # dropping label column and index
        # scaler.fit(self.trainingX)
        self.trainingY = self.training['ClaimAmount']  # getting only label column
        self.testingX = testset.drop(['rowIndex'], 1)  # drop row index
        self.no_claims = self.training[self.training['ClaimAmount'] == 0]  # get the samples with no claims
        self.claims = self.training[self.training['ClaimAmount'] != 0]  # get the samples with claims
        self.no_claim_X = self.no_claims.drop(['rowIndex', 'ClaimAmount'], 1)  # repeat above for this data-set
        self.no_claim_Y = self.no_claims['ClaimAmount']
        self.claim_X = self.claims.drop(['rowIndex', 'ClaimAmount'], 1)
        self.claim_Y = self.claims['ClaimAmount']
        self.mix_X = self.no_claim_X[0:7000]
        self.mix_X = self.mix_X.append(self.claim_X)
        self.mix_Y = self.no_claim_Y[0:7000]
        self.mix_Y = self.mix_Y.append(self.claim_Y)
        # print(self.trainingX)
        # self.trainingX = pd.DataFrame(data=scaler.transform(self.trainingX), columns=['feature1', 'feature2', 'feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11', 'feature12','feature13','feature14', 'feature15', 'feature16', 'feature17', 'feature18'])
        # print(self.trainingX)
        self.X_resampled, self.Y_resampled = ros.fit_resample(self.get_trainX_pd(), self.changeToBinary())

    def get_training(self):
        return self.training

    def get_osX(self):
        return self.X_resampled

    def get_osY(self):
        return self.Y_resampled

    def get_col_names(self):
        return list(self.trainingX)

    def get_trainx_by_feature(self, features):
        return np.array(self.trainingX.loc[:, features])

    def get_trainX(self):
        return np.array(self.trainingX)

    def get_trainY(self):
        return np.array(self.trainingY)

    def get_testX(self):
        return np.array(self.testingX)

    def get_trainX_pd(self):
        return self.trainingX

    def get_trainY_pd(self):
        return self.trainingY

    def get_testX_pd(self):
        return self.testingX

    def get_noClaimX_pd(self):
        return self.no_claim_X

    def get_noClaimY_pd(self):
        return self.no_claim_Y

    def get_claimX_pd(self):
        return self.claim_X

    def get_claimY_pd(self):
        return self.claim_Y

    def get_noClaimX(self):
        return np.array(self.get_noClaimX_pd())

    def get_noClaimY(self):
        return np.array(self.get_noClaimX_pd())

    def get_claimX(self):
        return np.array(self.get_claimX_pd())

    def get_claimY(self):
        return np.array(self.get_claimY_pd())

    def get_mixX(self):
        return self.mix_X

    def get_mixY(self):
        return self.mix_Y

    def changeToBinary(self):
        newArr = []
        for val in self.trainingY:
            newArr.append(0 if val == 0 else 1)
        return pd.DataFrame(data=newArr)


if __name__ == '__main__':
    test = DataSet()