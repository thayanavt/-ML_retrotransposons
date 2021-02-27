from dataset import Dataset
from model import Model 
from sklearn.neighbors import KNeighborsClassifier
import os

def train_model(dataset_dir):
    print(f'Training model from directory {dataset_dir}')
    data = Dataset(dataset_dir)
    X, y = data._get_set()
    parameters = dict(n_neighbors=list(range(1,17,2))) 

    model_knn = Model(X, y, KNeighborsClassifier(), parameters, "KNN PCA DAC")
    print(f'''Estimator: KNeighborsClassifier()
           Parameters: {parameters}
	   With PCA''')
    model_knn.standardization()
    model_knn.pca()
    print('Training the model')
    model_knn.train()
    print('Calculating metrics')
    model_knn.metrics_mean()

    model_knn = Model(X, y, KNeighborsClassifier(), parameters_knn, "KNN without PCA DAC")
    print(f'''Estimator: KNeighborsClassifier()
           Parameters: {parameters_knn}
	   Without PCA''')
    model_knn.standardization()
    print('Training the model')
    model_knn.train()
    print('Calculating metrics')
    model_knn.metrics_mean()

path = os.path.dirname(__file__) 
dataset_dir = f'{path}/datasets/dataset_DAC.csv'
train_model(dataset_dir)

#def _ask_model
