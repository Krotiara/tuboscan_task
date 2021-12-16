from App.DatasetLoader import DatasetLoader

if __name__ == '__main__':
    loader = DatasetLoader()
    x,y = loader.load_dataset('Files/dataset.csv')
    print(x[0])
    new_x = loader.random_vary_input_data(x, 10)
    print(new_x[0])


