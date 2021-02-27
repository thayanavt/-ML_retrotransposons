import numpy as np 
import pandas as pd

class Dataset:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
   
    def _get_set(self):   
        data = pd.read_csv(self.dataset_dir, header= None, low_memory = False)  
        X = data.iloc[1:, 1:-1].values   
        y = np.array(data.iloc[1:,-1])
        return X, y

 


