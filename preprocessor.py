# import numpy as np
import pandas as pd
LEAST_DISTINCT_VALUE = 20


def check_unique_values(data):
    for i in data:
        print(i)
        # print(np.unique(data[i]).size)


def one_hot_encode(raw_data, debug=False):
    new_data = pd.DataFrame({})
    for data in raw_data:
        col_type = raw_data[data].dtype
        col = raw_data[data]
        if col_type == 'object' or col.nunique() < LEAST_DISTINCT_VALUE:
            one_hot = pd.get_dummies(col, prefix=data)
            new_data = pd.concat([new_data, one_hot], axis=1)
        else:
            new_data = pd.concat([new_data, col], axis=1)
    if debug:
        new_data.info()
        print(new_data.shape)
    return new_data


if __name__ == '__main__':
    from load_csv import DataSet
    data = DataSet()
    check_unique_values(data.get_testX_pd())
    one_hot_encode(data.get_trainX_pd(), True)
