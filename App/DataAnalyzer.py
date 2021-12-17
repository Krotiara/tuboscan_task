import matplotlib.pyplot as plt
import seaborn as sb
import pandas
from pandas.plotting import scatter_matrix

class DataAnalyzer:

    def __init__(self):
        pass

    def analyse_data(self, data):
        pandas.set_option('display.max_columns', None)
        # print(data.dataframe.describe())
        self.box_plot(data)
        # self.plot_scatter_matrix(data)
        self.plot_params_histograms(data)
        self.plot_correlation_matrix(data)

    def plot_params_histograms(self, data):
        data.dataframe.hist()
        plt.show()

    def plot_correlation_matrix(self, data):
        plt.figure()
        heatmap = sb.heatmap(data.correlation_matrix, cmap="coolwarm",
                             vmin=-1, vmax=1, annot=True, linewidths=1,
                             linecolor='black')
        heatmap.set_title('Correlation Heatmap')
        plt.show()

    def box_plot(self, data):
        plot = data.dataframe.boxplot()
        plot.set_title('box plot')
        plt.show()

    def plot_scatter_matrix(self, data):
        scatter_matrix(data.dataframe, diagonal="kde")
        plt.show()
