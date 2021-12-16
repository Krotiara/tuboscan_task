import numpy as np
from App.PreProcessedData import PreProcessedData


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
        dataframe = self._normalize_data(dataframe)
        train_X, train_Y = self._convert_dataframe_to_sets(dataframe)
        return PreProcessedData(dataframe,missing_rows_count, train_X, train_Y)

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
    def _convert_dataframe_to_sets(dataframe):
        """
        Преобразовать датасет в набор тренируемых и контрольных значений в
        соответсвии с постановкой задания.
        :param dataframe: pandas dataframe
        :return: train_x and train_y sets
        """
        float_data = dataframe.to_numpy()
        x, y = np.hsplit(float_data, [8])
        return x, y
