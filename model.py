from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, accuracy_score, f1_score
import numpy as np 
import statistics
import os


class Model:
    def __init__(self, X, y, classifier, parameters, name_model):   
        self.X = X
        self.y = y     
        self.classifier = classifier
        self.parameters = parameters 
        self.name_model = name_model
        self.precisions = []
        self.recalls = []
        self.accuracies = [] 
        self.f1_scores = []

    def standardization(self):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def pca(self, graph=0): 
        pca = PCA(n_components=6).fit(self.X)
        percent_variance = list(np.round(pca.explained_variance_ratio_, decimals = 2))
        percent_variance_dict ={}

        for i,j in zip(range(1,7), percent_variance):
            percent_variance_dict[i] = j

        self.sorted_keys= sorted(percent_variance_dict, key = percent_variance_dict.get)

        pcs = PCA().fit_transform(self.X)
        first_principalcomp = pcs[:,(self.sorted_keys[-1]-1)]
        second_principalcomp = pcs[:,(self.sorted_keys[-2]-1)]

        principalcomps = np.array([first_principalcomp, second_principalcomp]).transpose()
        self.X = principalcomps
        
    def train(self):
        kf = KFold(n_splits=10, shuffle=True, random_state=7) 
        for train_index, test_index in kf.split(self.X, y=self.y):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            clf = GridSearchCV(self.classifier, self.parameters, cv=10)
            clf.fit(X_train, y_train) 
            y_pred = clf.predict(X_test)
            
            precision = self.precisions.append(precision_score(y_test, y_pred, average = 'macro', zero_division= 1))
            recall = self.recalls.append(recall_score(y_test, y_pred, average = 'macro', zero_division= 1))
            accuracy = self.accuracies.append(accuracy_score(y_test, y_pred))
            f1 = self.f1_scores.append(f1_score(y_test, y_pred, average = 'macro'))
    
    def metrics_mean(self):
        m_precisions = sum(self.precisions)/len(self.precisions)
        std_precisions = statistics.stdev(self.precisions)

        m_recalls = sum(self.recalls)/len(self.recalls)
        std_recalls = statistics.stdev(self.recalls)

        m_accuracies = sum(self.accuracies)/len(self.accuracies)
        std_accuracies = statistics.stdev(self.accuracies)   
        
        m_f1_scores = sum(self.f1_scores)/len(self.f1_scores)
        std_f1_scores = statistics.stdev(self.f1_scores) 

        print("Saving metrics to a file")
        
        path = os.path.dirname(__file__)
        with open(f"{path}/metrics.csv", 'a') as file_metrics:
            file_metrics.write(f'''{self.name_model},Precision,{m_precisions},{std_precisions}
        {self.name_model},Recall,{m_recalls},{std_recalls}
        {self.name_model},Accuracy,{m_accuracies},{std_accuracies}
        {self.name_model},F1-score,{m_f1_scores},{std_f1_scores}''')
        
    #def serialize(self):
    #desetialize@
