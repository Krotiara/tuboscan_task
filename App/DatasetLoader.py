import random
import numpy as np
import pandas as pd


class DatasetLoader:

    def __init__(self):
        pass

    def load_dataset(self, path_to_csv):
        """Загрузка датасета из файла CSV в формате тестового задания.
         Функция Возвращает 2 массива, соответствующие входным данным
         колонок A-C и предсказательными значениями колонок D-F соответственно.
         """
        dataframe = pd.read_csv(path_to_csv, delimiter=',')
        float_data = dataframe.to_numpy().astype(np.float)
        x, y = np.hsplit(float_data, [8])
        return x, y

    def random_vary_input_data(self, input_data, vary_limit):
        """Случайно проварьировать входные данные input_data в пределах
        vary_limit процентов от их значений. Функция возвращает
         скорректированный набор input_data
        input_data - numpy array,
        vary_limit - числовое значение в процентах.
        """
        random_vary = lambda x: self._random_vary_value(x, vary_limit)
        varied_data = random_vary(input_data)
        return varied_data

    def _random_vary_value(self, value, vary_limit):
        percent_value = value / 100 * vary_limit
        return random.uniform(value - percent_value, value + percent_value)
