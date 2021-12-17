import numpy as np
from App.PreProcessedData import PreProcessedData
from scipy import  stats


class DataPreProcessor:
    def __init__(self):
        pass

    def preprocess_data(self, dataframe):
        """
        Предварительно обработать датасет.
        :param dataframe: pandas dataframe
        :return: Объект класса PreProcessedData
        """
        missing_rows_count = self._process_missing_data_by_delete(dataframe)
        dataframe = self._remove_outliers_iqr(dataframe)
        dataframe = self._normalize_data(dataframe)
        return PreProcessedData(dataframe, missing_rows_count)

    @staticmethod
    def _process_missing_data_by_delete(dataframe):
        """
        Функция обрабатывает пропущенные в датасете значения путем их
        удаления.
        :param dataframe: pandas dataframe
        :return: Количество обработанных строк с
        пропущенными значениями.
        """
        # Drop the rows where at least one element is missing.
        missing_rows_count = sum([1 for i, row in dataframe.iterrows()
                                  if any(row.isnull())])
        dataframe.dropna(inplace=True)
        return missing_rows_count

    @staticmethod
    def _normalize_data(dataframe):
        return dataframe / dataframe.max()

    @staticmethod
    def _remove_outliers_z_score(dataframe):
        z_scores = stats.zscore(dataframe)
        print(dataframe.shape)

        abs_z_scores = np.abs(z_scores)
        # mask to remove outliers (outlier ~ z score >= 3)
        mask = (abs_z_scores < 3).all(axis=1)
        dataframe_without_outliers = dataframe[mask]
        print(dataframe_without_outliers.shape)
        return  dataframe_without_outliers

    @staticmethod
    def _remove_outliers_iqr(dataframe):
        print(dataframe.shape)
        Q1 = dataframe.quantile(0.25)
        Q3 = dataframe.quantile(0.75)
        IQR = Q3-Q1
        lower_range = Q1 - (1.5 * IQR)
        upper_range = Q3 + (1.5 * IQR)
        mask = ((dataframe < lower_range) | (dataframe > upper_range)).any(axis=1)
        dataframe_without_outliers = dataframe[~mask]
        print(dataframe_without_outliers.shape)
        return dataframe_without_outliers







