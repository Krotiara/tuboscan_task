from App.DatasetLoader import DatasetLoader
from App.DataAnalyzer import DataAnalyzer
from App.DataPreProcessor import DataPreProcessor

if __name__ == '__main__':
    loader = DatasetLoader()
    analyzer = DataAnalyzer()
    preprocessor = DataPreProcessor()
    dataframe = loader.load_dataset('Files/dataset.csv')
    preprocessedData = preprocessor.preprocess_data(dataframe)
    analyzer.analyse_data(preprocessedData.dataframe)



