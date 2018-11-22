import os
import numpy as np


class FileWriter:
    def __init__(self, file_name='test', data=[1,2,3]):
        self.col_names = ['rowIndex', 'ClaimAmount']
        self.file = os.path.join('./predictions/', file_name + '.csv')
        self.data = np.array(data)

    def write(self):
        with open(self.file, 'w') as file:
            file.write('rowIndex,ClaimAmount')
            for i in range(len(self.data)):
                line = '\n{},{}'.format(i, self.data[i])
                file.write(line)
        file.close()
        return


if __name__ == '__main__':
    f = FileWriter()
    f.write()
