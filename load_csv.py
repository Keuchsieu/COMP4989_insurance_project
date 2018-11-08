import pandas as pd
import numpy as np


class DataSet:
    def __init__(self):
        training = pd.read_csv('./datasets/trainingset.csv')
        testset = pd.read_csv('./datasets/testset.csv')
        self.trainingX = training.drop(['rowIndex', 'ClaimAmount'], 1)  # dropping label column and index
        self.trainingY = training['ClaimAmount']  # getting only label column
        self.testingX = testset.drop(['rowIndex'], 1)  # drop row index
        self.no_claim_X = 0
        self.no_claim_Y = 0
        self.claim_X = 0
        self.claim_Y = 0

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


if __name__ == '__main__':
    test = DataSet()
    print(test.get_trainX().shape)
    print(test.get_trainY().shape)
    print(test.get_testX().shape)
