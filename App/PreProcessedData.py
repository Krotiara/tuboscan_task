import numpy as np


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
        return x,

    @property
    def correlation_matrix(self):
        return self.dataframe.corr(method='pearson')
