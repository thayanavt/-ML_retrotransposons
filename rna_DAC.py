from dataset import Dataset
from model import Model 
from sklearn.neural_network import MLPClassifier
import os

def train_model(dataset_dir):
    print(f'Training model from directory {dataset_dir}')
    data = Dataset(dataset_dir)
    X, y = data._get_set()
    parameters = {'batch_size':[1,50,100]}
    model_RNA = Model(X, y, MLPClassifier(hidden_layer_sizes=(174, ),early_stopping = True, max_iter = 1000), parameters, "Neural Network DAC")
    print(f'''Estimator: MLPClassifier()
           Parameters: {parameters}
	   Without PCA''')
    model_RNA.standardization()
    print('Training the model')
    model_RNA.train()
    print('Calculating metrics')
    model_RNA.metrics_mean()

path = os.path.dirname(__file__) 
dataset_dir = f'{path}/datasets/dataset_DAC.csv'
train_model(dataset_dir)

#def _ask_model
