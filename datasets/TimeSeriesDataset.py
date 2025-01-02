import torch
import numpy as np

class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    Dataset para séries temporais.
    
    Esta classe herda de torch.utils.data.Dataset e implementa um dataset personalizado para trabalhar com séries temporais.
    O dataset processa as séries temporais aplicando z-normalização e criando rótulos para cada série.
    """
    def __init__(self, tsv_file):        
        """
        Inicializa o dataset.
        
        Parâmetros
        ----------
        tsv_file : str
            Caminho para o arquivo TSV contendo os dados das séries temporais
        """
        with open(tsv_file, "r") as file:
            self.data = file.readlines()            
        
        time_series = np.array([np.array(d.split("\t")[1:], dtype=np.float32) for d in self.data])
        
        #apply z-normalization        
        
        time_series[np.isnan(time_series)] = 0
        std_ = time_series.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        time_series = (time_series - time_series.mean(axis=1, keepdims=True)) / std_        
        
        self.time_series = time_series
        labels = np.array([int(line.split("\t")[0]) for line in self.data])
        self.labels = torch.tensor(np.array([sorted(list(set(labels))).index(l) for l in labels]))
    
    @property
    def num_classes(self):
        return len(np.unique(self.labels))
    
    @property
    def num_features(self):
        return self.time_series.shape[1]

    def __getitem__(self, idx):
        return self.time_series[idx], self.labels[idx]

    def __len__(self):
        return len(self.time_series)   

