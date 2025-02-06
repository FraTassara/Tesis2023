import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import utils
from utils import to_categorical

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = MinMaxScaler()
        
    def load_and_preprocess(self):
        # Cargar y procesar datos
        dataset = pd.read_csv(self.file_path)
        dataset = dataset.drop(['ID','SEXO', 'DISP1', 'DISP2', 'DISP3', 'DISP4', 'DISP5', 
                              'TVIA3', 'TCAM3', 'TESP3', 'CTOT3_w', 'JEFH', 'HTS'], axis=1)
        dataset = dataset[dataset["ICH"] > 0]
        dataset = dataset[dataset["ICH"].isin([1, 2, 4, 5])]
        
        X = dataset.loc[:, dataset.columns != 'ICH']
        y = dataset['ICH'].replace({1:0, 2:1, 4:2, 5:3})
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X.astype('float32'))
        
        # Codificar etiquetas
        y_cat = to_categorical(y)
        
        return X_scaled, y_cat, self.scaler, X.columns