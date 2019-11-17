'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-10-10 21:57:24
@LastEditors: Jack Huang
@LastEditTime: 2019-10-10 22:34:46
'''

import pandas as pd 
import os 
import matplotlib.pyplot as plt 

class logger:
    def __init__(self, logger_name, logger_path, col_names, restore=False):
        self.logger_name = logger_name
        if not os.path.exists(logger_path):
            os.makedirs(logger_path)
        self.logger_path = logger_path
        self.col_names = col_names
        self.save_path = os.path.join(self.logger_path, self.logger_name + '.csv')
        if restore:
            self.dataLogs = self.load()
        else:
            self.dataLogs = pd.DataFrame(columns=col_names) 

    def add_row_data(self, one_row, saveFlag=False):
        last_index = len(self.dataLogs)
        self.dataLogs.loc[last_index] = one_row
        if saveFlag:
            self.save()

    def save(self):
        self.dataLogs.to_csv(self.save_path)

    def load(self):
        dataLogs_Saved = pd.read_csv(self.save_path)
        return dataLogs_Saved

    def plotToFile(self, title, showFlag = False):
        plt.clf()
        plt.title(title)
        ax = plt.gca()
        self.dataLogs.plot(kind='line', x = self.col_names[0], y=self.col_names[1], ax=ax)
        self.dataLogs.plot(kind='line', x = self.col_names[0], y=self.col_names[2], ax=ax)
        self.dataLogs.plot(kind='line', x = self.col_names[0], y=self.col_names[3], ax=ax)
        self.dataLogs.plot(kind='line', x = self.col_names[0], y=self.col_names[4], ax=ax)
        plt.savefig(os.path.join(self.logger_path, self.logger_name + '.png'))
        if showFlag:
            plt.show()

    def close(self):
        self.save()