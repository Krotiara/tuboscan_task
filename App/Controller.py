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

    def load_data_set(self, path, output_columns_index):
        dataframe = self.loader.load_dataset(path)
        self.datahandler = DataHandler(dataframe, output_columns_index)


    def predict(self, input_data):
        predict_values = self.predictor.predict(input_data)
        return self.datahandler.denormalize_output(predict_values)
    

