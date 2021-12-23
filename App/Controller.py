from App.DatasetLoader import DatasetLoader
from App.DataAnalyzer import DataAnalyzer
from App.DataHandler import DataHandler
from App.Predictor import Predictor

class Controller:
    def __init__(self):
        self.loader = DatasetLoader()
        self.analyzer = DataAnalyzer()
        self.predictor = Predictor()
        self.datahandler = None

    def load_data_set(self, path):
        dataframe = self.loader.load_dataset(path)
        self.datahandler = DataHandler(dataframe, ['D', 'E', 'F'])


    def predict(self, input_data):
        predict_values = self.predictor.predict(input_data)
        return self.datahandler.denormalize_output(predict_values)

    def get_condidence_interval_description(self, r2_score, shape_x, p_value):
        interval = self.predictor.calc_error_confidence_interval(r2_score,
         shape_x, p_value)
        return "{0}% доверительный интервал для r^2 = {1:.4f} +/- {2:.4f}" \
         .format(p_value*100, r2_score, interval)
    

