import pandas as pd
import numpy as np


class DataSet:
    def __init__(self):
        self.training = pd.read_csv('./datasets/trainingset.csv')
        testset = pd.read_csv('./datasets/testset.csv')
        self.trainingX = self.training.drop(['rowIndex', 'ClaimAmount'], 1)  # dropping label column and index
        self.trainingY = self.training['ClaimAmount']  # getting only label column
        self.testingX = testset.drop(['rowIndex'], 1)  # drop row index
        self.no_claims = self.training[self.training['ClaimAmount'] == 0] # get the samples with no claims
        self.claims = self.training[self.training['ClaimAmount'] != 0] # get the samples with claims
        self.no_claim_X = self.no_claims.drop(['rowIndex', 'ClaimAmount'], 1) # repeat above for this data-set
        self.no_claim_Y = self.no_claims['ClaimAmount']
        self.claim_X = self.claims.drop(['rowIndex', 'ClaimAmount'], 1)
        self.claim_Y = self.claims['ClaimAmount']

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


if __name__ == '__main__':
    test = DataSet()
    print(test.get_trainX().shape)
    print(test.get_trainY().shape)
    print(test.get_testX().shape)
