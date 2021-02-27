from dataset import Dataset
from model import Model 
from sklearn.tree import DecisionTreeClassifier
import numpy as np 
import os

def train_model(dataset_dir):
    print(f'Training model from directory {dataset_dir}')
    data = Dataset(dataset_dir)
    X, y = data._get_set()
    parameters = {'criterion':['entropy'], 'max_depth': np.arange(3, 12)}
    model_decisionTree = Model(X, y, DecisionTreeClassifier(), parameters, "Decision Tree DAC")
    print(f'''Estimator: DecisionTreeClassifier()
           Parameters: {parameters}
	   Without PCA''')
    model_decisionTree.standardization()
    print('Training the model')
    model_decisionTree.train()
    print('Calculating metrics')
    model_decisionTree.metrics_mean()

path = os.path.dirname(__file__) 
dataset_dir = f'{path}/datasets/dataset_DAC.csv'
train_model(dataset_dir)

#def _ask_model

