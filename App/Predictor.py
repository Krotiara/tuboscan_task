import numpy as np
from numpy.core.fromnumeric import mean
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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
        self.fit_mean_errors = \
        self._calculate_fit_mean_abs_errors(x_train, y_train)

    def predict(self, x_predict):       
        y_predict = self.model.predict(x_predict)
        return y_predict

    def _calculate_fit_mean_abs_errors(self, x_train, y_train):
        predictions = self.predict(x_train)
        mean_errors = mean_squared_error(y_train, predictions,
            multioutput='raw_values',squared=False)
        return mean_errors


