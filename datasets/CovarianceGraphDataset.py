import torch
import os
import os.path as osp
import numpy as np
import dgl
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info
import traceback
from sim_tsc import distance_matrix
import networkx as nx
import utils

class CovarianceGraphDataset(dgl.data.DGLDataset):
    """
    Dataset para grafos de covariância.
    
    Esta classe herda de DGLDataset e implementa um dataset personalizado para trabalhar com 
    grafos de covariância construídos a partir de séries temporais. O dataset processa as séries
    temporais calculando a matriz de correlação de Pearson entre elas e criando arestas quando
    a correlação absoluta excede um limiar R.

    Parâmetros
    ----------
    root : str
        Caminho raiz onde os dados serão salvos
    tsv_file : str
        Arquivo TSV contendo os dados de entrada
    R : float
        Limiar de correlação para criar arestas no grafo. Apenas correlações com valor
        absoluto maior que R geram arestas.

    Atributos
    ----------
    root : str
        Diretório raiz do dataset
    data_type : str
        Tipo de dados (processado)
    save_path : str
        Caminho onde os dados processados são salvos
    labels : torch.Tensor
        Rótulos das classes
    num_classes : int
        Número de classes únicas
    num_features : int
        Número de características dos nós (comprimento das séries temporais)
    R : float
        Limiar de correlação usado
    graph : dgl.DGLGraph
        Grafo DGL construído
    """
    def __init__(self, root, tsv_file, R,):
        print("Creating dataset...")
        self.root = root
        self.data_type = "processed"
        
        graph_info = f"R_{R}"
        
        self.save_path = osp.join(self.root, graph_info, self.data_type)

        with open(tsv_file, "r") as file:
            self.__train_data = file.readlines()
        
        labels = np.array([int(line.split("\t")[0]) for line in self.__train_data])
        self.labels = torch.tensor(np.array([sorted(list(set(labels))).index(l) for l in labels]))
        
        self.num_classes = len(np.unique(self.labels))
        self.R = R

        super().__init__(
            name="CovarianceGraphDataset",
            raw_dir=root,
            save_dir=self.save_path,
            force_reload=False,
            verbose=False,
        )
        print("Dataset created!")

    def download(self):
        pass

    @property
    def save_path(self):
        return self.__save_path

    @save_path.setter
    def save_path(self, value):
        if not os.path.exists(value):
            os.makedirs(value)
        self.__save_path = value

    @property
    def data_type(self):
        return self.__data_type

    @data_type.setter
    def data_type(self, value):
        self.__data_type = value

    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, value):
        self.__labels = value

    @property
    def num_classes(self):
        return self.__num_classes

    @num_classes.setter
    def num_classes(self, value):
        self.__num_classes = value
        
    @property
    def num_features(self):
        return self.__num_features
    
    @num_features.setter
    def num_features(self, value):
        self.__num_features = value

    def process(self):
        """
        Processa os dados brutos e cria um grafo DGL baseado na matriz de correlação de Pearson.
        
        Este método:
        1. Carrega as séries temporais do arquivo TSV
        2. Calcula a matriz de correlação de Pearson entre as séries
        3. Aplica um limiar R para binarizar as correlações
        4. Converte a matriz de adjacência em um grafo DGL
        5. Adiciona rótulos e características aos nós
        
        O grafo resultante contém:
        - Nós representando as séries temporais individuais
        - Arestas entre séries com correlação >= R
        - Características dos nós sendo as próprias séries temporais
        - Rótulos das classes como tensores
        """
        print("Processing dataset...")     
        time_series = np.array([np.array(line.split("\t")[1:], dtype=float) for line in self.__train_data])
        distances = distance_matrix.create_pearson_matrix(time_series)        
        adj = np.absolute(distances)
        
        #if the correlation is below the threshold R, remove the edge, otherwise keep the edge        
        adj[adj < self.R] = 0
        adj[adj >= self.R] = 1
        
        dgl_graph = dgl.from_networkx(nx.from_numpy_array(adj))   
        dgl_graph.ndata['label'] = torch.tensor(self.labels, dtype=torch.long).view(-1, 1)
        dgl_graph.ndata['feat'] = torch.tensor(time_series, dtype=torch.float32)        
        
        self.graph = dgl_graph
        self.num_features = time_series.shape[-1]
    def __getitem__(self,idx):
        return self.graph

    def __len__(self):
        return 1

    def save(self):
        print(f"Saving processed data in {self.save_path}")
        graph_path = os.path.join(self.save_path, "_dgl_graph.bin")
        try:
            save_graphs(graph_path, self.graph)
            info_path = os.path.join(self.save_path, "_info.pkl")
            save_info(
                info_path,
                {
                    "num_classes": self.num_classes,                    
                    "num_features": self.num_features,
                    "R" : self.R,                    
                },
            )
        except:
            print("Error saving processed data.")
            print(traceback.format_exc())
            return
        print("Data saved!")

    def load(self):
        print(f"Loading processed data from {self.save_path}")

        graph_path = os.path.join(self.save_path, "_dgl_graph.bin")

        try:
            self.graph, _ = load_graphs(graph_path)
            self.graph = self.graph[0]
            info_path = os.path.join(self.save_path, "_info.pkl")
            
            self.num_classes = load_info(info_path)["num_classes"]            
            self.R = load_info(info_path)["R"]
            self.num_features = load_info(info_path)["num_features"]
            
        except:
            print("Error loading processed data. Try to process the data again.")
            print(traceback.format_exc())
            return

        print(f"Data loaded from {self.save_path}")

    def has_cache(self):
        # Check whether there is processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, "_dgl_graph.bin")
        info_path = os.path.join(self.save_path, "_info.pkl")

        if os.path.exists(graph_path) and os.path.exists(info_path):
            print(f"Found processed data in {self.save_path}")
            return True
        else:
            print(f"Processed data not found in {self.save_path}")
            return False

    def plot_graph(self):
        utils.plot_graph(self.graph)()