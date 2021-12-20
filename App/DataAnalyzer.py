import matplotlib.pyplot as plt
import seaborn as sb
import pandas
from pandas.plotting import scatter_matrix


class DataAnalyzer:

    def __init__(self):
        pass
    
    def describe_data(self, data):
        return data.describe

    def plot_params_histograms(self, data):
        data.hist(bins = 30)
        plt.show()

    def plot_compare_hist_and_distribution(self, paramData):
        sb.distplot(paramData, bins=40)
        plt.show()

    def plot_correlation_matrix(self, correlation_matrix):
        fig, ax = plt.subplots(figsize=(10,10))
        heatmap = sb.heatmap(correlation_matrix, cmap="coolwarm",
                             vmin=-1, vmax=1, annot=True, linewidths=0.5,
                             linecolor='black', ax=ax)
        heatmap.set_title('Correlation Heatmap')
        plt.show()

    def plot_p_values_matrix(self, matrix_dataframe):
        fig, ax = plt.subplots(figsize=(10,10))
        sb.heatmap(matrix_dataframe, ax=ax, annot = True, cbar=False,
        cmap= 'coolwarm')
        ax.set_title('P values matrix')
        plt.show()

    def box_plot(self, data):
        plot = data.boxplot()
        plot.set_title('box plot')
        plt.show()

    def plot_scatter_matrix(self, data):
        fig, ax = plt.subplots(figsize=(4,4))
        scatter_matrix(data, diagonal="kde", ax=ax)
        plt.show()

    def plot_correlation_statistic(self, correlation_matrix, p_values_matrix):
        table_data, table_labels = \
         self._get_correlation_statistic(correlation_matrix, p_values_matrix)    
        fig, ax =plt.subplots(figsize=(10,10))
        ax.xaxis.set_visible(False) 
        ax.yaxis.set_visible(False)
        table = ax.table(cellText=table_data,colLabels=table_labels,
        loc="center", cellLoc='center')
        table.set_fontsize(14)
        table.scale(2, 2)
        plt.show()

    def profile_data(self, data):
        import pandas_profiling
        profile = pandas_profiling.ProfileReport(data)
        #fix problem between matplotlib and pandas_profiling
        plt.style.use('default') 
        return profile

    def _get_correlation_statistic(self, correlation_matrix, p_values_matrix):
        c_m_values = correlation_matrix.values
        p_v_values = p_values_matrix.values
        shape = correlation_matrix.shape
        columns_names = correlation_matrix.columns
        table_labels = ["Param 1 - Param 2", "Correlation coef (Power)",
        "P value (Certainty)"]
        table_data = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i >= j:
                    continue             
                params = "{0}-{1}:" \
                .format(columns_names[i],columns_names[j])
                row = [params,
                self._get_correlation_description(c_m_values[i,j]),
                self._get_P_value_description(p_v_values[i,j])]
                table_data.append(row)
                #table_data = np.append(table_data, row, axis=0)    
        return table_data, table_labels

    @staticmethod
    def _get_P_value_description(p_value):
        certainty = ""
        if p_value < 0.001:
            certainty = "Strong"
        elif p_value < 0.05:
            certainty = "Moderate"
        elif p_value < 0.1:
            certainty = "Weak"
        elif p_value > 0.1:
             certainty = "No"
        return "{0} ({1})"\
        .format(certainty, round(p_value,3))

    @staticmethod
    def _get_correlation_description(correlation_coef):
        power = ""
        correlation_coef = abs(correlation_coef)
        if correlation_coef > 0.9:
            power = "Very high"
        elif correlation_coef > 0.7:
            power = "High"
        elif correlation_coef > 0.5:
            power = "Moderate"
        elif correlation_coef > 0.2:
            power = "Low"
        elif correlation_coef < 0.2:
            power = "Very Low"

        return "{0} ({1})".format(power,round(correlation_coef,3))





