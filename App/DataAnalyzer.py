import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas
from sklearn import preprocessing
import numpy as np
import random


class DataAnalyzer:

    def __init__(self):
        pass

    def analyse_data(self, dataframe):
        pandas.set_option('display.max_columns', None)
        print(dataframe.describe())
        self._plot_params_histograms(dataframe)
        plt.show()

    def _plot_params_histograms(self, dataframe):
       dataframe.hist()


