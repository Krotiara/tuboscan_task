import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from scipy import  stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



class DataHandler:

    def __init__(self, dataframe, target_columns):
        self.dataframe = dataframe
        self.processed_missing_rows_count = 0
        self.removed_duplicates_count = 0
        self.delete_outliers_count = 0
        self.target_columns = target_columns
        self.input_scaler = MinMaxScaler(feature_range=(0,1))
        self.output_scaler = MinMaxScaler(feature_range=(0,1))

    @property
    def pre_processed_statistic(self):
        return "Processed missing rows count: {0},\
        \nRemoved duplicates count: {1}.,\n" \
        "Removed outliers count: {2}"\
        .format(self.processed_missing_rows_count,
        self.removed_duplicates_count, self.delete_outliers_count)
    
    def split_data_into_train_and_testing_sets(self):
        x_data = self.dataframe.drop(columns=self.target_columns)
        y_data = pd.DataFrame(self.dataframe[self.target_columns])
        X, X_test, Y, Y_test = train_test_split(x_data,y_data,test_size=0.3,
                                                random_state=42)
        return X, X_test, Y, Y_test

    @property
    def correlation_data(self):
        return self.get_correlation_data(self.dataframe)

    def get_correlation_data(self, dataframe):
        """
        Получить матрицы коэффициентов корреляции и уровней значимости.
        :return: матрицы коэффициентов корреляции и уровней значимости.
        """
        values = dataframe.values.T
        columns_len = len(dataframe.columns)
        correlation_matrix = np.empty((columns_len,columns_len), dtype=float)
        p_values_matrix = np.empty((columns_len,columns_len), dtype=float)
        for i, paramData1 in enumerate(values):
            for j, paramData2 in enumerate(values):
                if i > j:
                    continue #симметричность матриц
                corr = pearsonr(paramData1,paramData2)
                correlation_matrix[i,j] = corr[0]
                correlation_matrix[j,i] = corr[0]
                p_values_matrix[i,j] = corr[1]
                p_values_matrix[j,i] = corr[1]
        df_corr_matrix = pd.DataFrame(
            data = correlation_matrix,
            columns=dataframe.columns,
            index=dataframe.columns)
        df_p_vals = pd.DataFrame(
            data = p_values_matrix, 
            columns=dataframe.columns, 
            index=dataframe.columns)
        return df_corr_matrix, df_p_vals

    @property
    def kendall_correlation_matrix(self):
        return self.dataframe.corr(method='kendall')

    def preprocess_data(self):
        """
        Предварительно обработать датасет.
        """
        self.missing_rows_count = self._process_missing_data_by_delete()
        #self.dataframe, self.delete_outliers_count = self._remove_outliers_iqr()
        self.duplicates_count = self._remove_duplicates()
        self.dataframe = self._normalize_dataframe() 

    def _normalize_dataframe(self):
        x = self.dataframe.drop(columns=self.target_columns)
        y = pd.DataFrame(self.dataframe[self.target_columns])
        normalized_x = self.input_scaler.fit_transform(x)
        normalized_y = self.output_scaler.fit_transform(y)
        dataframe = pd.DataFrame(np.column_stack([normalized_x, normalized_y]),
        columns=self.dataframe.columns,
        index=self.dataframe.index) 
        return dataframe
       
    def normalize_input(self,data):
        return self.input_scaler.transform(data)
    
    def denormalize_output(self, data):
        return self.output_scaler.inverse_transform(data)

    def _process_missing_data_by_delete(self):
        """
        Функция обрабатывает пропущенные в датасете значения путем их
        удаления.
        :param dataframe: pandas dataframe
        :return: Количество обработанных строк с
        пропущенными значениями.
        """
        # Drop the rows where at least one element is missing.
        missing_rows_count = sum([1 for i, row in self.dataframe.iterrows()
                                  if any(row.isnull())])
        self.dataframe.dropna(inplace=True)
        return missing_rows_count
   
    def _remove_duplicates(self):
        shape = self.dataframe.shape[0]
        # drop float duplicates with less precision
        dataframe = \
        self.dataframe.loc[self.dataframe.round(4).drop_duplicates().index]
        duplicates_count = dataframe.shape[0] - shape
        return dataframe, duplicates_count
  
    def _remove_outliers_z_score(self):
        z_scores = stats.zscore(self.dataframe)
        abs_z_scores = np.abs(z_scores)
        # mask to remove outliers (outlier ~ z score >= 3)
        mask = (abs_z_scores < 3).all(axis=1)
        dataframe_without_outliers = self.dataframe[mask]
        return  dataframe_without_outliers

    def _remove_outliers_iqr(self): 
        Q1 = self.dataframe.quantile(0.25)
        Q3 = self.dataframe.quantile(0.75)
        IQR = Q3-Q1
        # 1.5 ~ inner fence, 3 ~ outer fence ~ extreme outlier
        lower_range = Q1 - (3 * IQR) 
        upper_range = Q3 + (3 * IQR)
        mask = ((self.dataframe < lower_range) |
         (self.dataframe > upper_range)).any(axis=1)
        dataframe_without_outliers = self.dataframe[~mask]
        delete_outliers_count = \
        self.dataframe.shape[0] - dataframe_without_outliers.shape[0]
        return dataframe_without_outliers, delete_outliers_count

    

    


