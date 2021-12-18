import numpy as np
from scipy.stats import pearsonr
import pandas as pd


class PreProcessedData:

    def __init__(self, dataframe, missing_rows_count):
        self.dataframe = dataframe
        self.processed_missing_rows_count = missing_rows_count

    @property
    def training_sets(self):
        """
        Преобразовать датасет в набор тренируемых и контрольных значений в
        соответсвии с постановкой задания.
        :param dataframe: pandas dataframe
        :return: train_x and train_y sets
        """
        float_data = self.dataframe.to_numpy()
        x, y = np.hsplit(float_data, [8])
        return x, y

    @property
    def correlation_data(self):
        values = self.dataframe.values.T
        columns_len = len(self.dataframe.columns)
        correlation_matrix = np.empty((columns_len,columns_len), dtype=float)
        p_values_matrix = np.empty((columns_len,columns_len), dtype=float)
        for i, paramData1 in enumerate(values):
            for j, paramData2 in enumerate(values):
                if i > j:
                    continue #симметричность матриц
                corr = pearsonr(paramData1,paramData2)
                correlation_matrix[i,j] = corr[0]
                correlation_matrix[j,i] = corr[0]
                p_values_matrix[i,j] = corr[1]
                p_values_matrix[j,i] = corr[1]
        df_corr_matrix = pd.DataFrame(
            data = correlation_matrix,
            columns=self.dataframe.columns,
            index=self.dataframe.columns)
        df_p_vals = pd.DataFrame(
            data = p_values_matrix, 
            columns=self.dataframe.columns, 
            index=self.dataframe.columns)
        return df_corr_matrix, df_p_vals


       # return self.dataframe.corr(method='pearson')


    @property
    def kendall_correlation_matrix(self):
        return self.dataframe.corr(method='kendall')
