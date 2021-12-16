class PreProcessedData:

    def __init__(self, dataframe, missing_rows_count, train_X, train_Y):
        self.dataframe = dataframe
        self.processed_missing_rows_count = missing_rows_count
        self.train_X, self.train_Y = train_X, train_Y
