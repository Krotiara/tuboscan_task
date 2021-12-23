import numpy as np
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import average
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as ss

class Predictor:
    def __init__(self):  
        self.model = LinearRegression()
        self.fit_mean_errors = None
        pass

    def get_fit_mean_errors(self):
        return self.fit_mean_errors
 
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_predict):       
        y_predict = self.model.predict(x_predict)
        return y_predict

    def calc_mean_abs_error(self, y_true, y_predict):
        return mean_absolute_error(y_true, y_predict)

    def calc_mean_squared_error(self, y_true, y_predict):
        '''
        Возвращает RMSE
        '''
        return mean_squared_error(y_true,y_predict, squared=True,
         multioutput='raw_values')

    def calc_r2_score(self, y_true, y_predict):
        return r2_score(y_true,y_predict)

    def calc_error_confidence_interval(self, value, n, p_value):
        '''
        value - оцениваемое значение,
        n - размер выборки,
        p_value - доверительная вероятность (в формате 0.xx).
        return: Доверительный интервал.
        '''
        #number of standard deviations from the Gaussian distribution
        z = ss.norm.ppf(p_value)
        interval = z * np.sqrt( (value * (1 - value)) / n)
        return interval


