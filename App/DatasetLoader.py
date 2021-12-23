import pandas as pd
import random


class DatasetLoader:

    @staticmethod
    def load_dataset(path_to_csv: str):
        """Загрузить датасет в тестовом формате,
         все колонки которого имеют численные значения"""
        dataframe = pd.read_csv(path_to_csv, delimiter=',')
        float_dataframe = dataframe.astype(float)
        return float_dataframe

    
    def random_vary_input_data(self, input_data, vary_limit: int):
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
