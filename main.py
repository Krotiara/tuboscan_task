from App.Controller import Controller

if __name__ == '__main__':
    controller = Controller()
    controller.load_data_set('Files/dataset.csv',8)
    controller.datahandler.preprocess_data()
    controller.analyzer.plot_scatter_matrix(controller.datahandler.dataframe)



