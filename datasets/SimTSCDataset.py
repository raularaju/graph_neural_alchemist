import torch
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import dgl
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info
import traceback
from sim_tsc import distance_matrix
import matplotlib.pyplot as plt
import networkx as nx

class SimTSCDataset(dgl.data.DGLDataset):
    """
    Dataset personalizado para manipular dados de séries temporais convertidos em grafos usando similaridade DTW.

    Esta classe implementa um dataset que:
    1. Recebe séries temporais como entrada
    2. Calcula a matriz de distância DTW entre as séries
    3. Constrói um grafo baseado nas K séries mais similares
    4. Armazena o grafo resultante no formato DGL

    Args:
        root (str): Diretório raiz para armazenar os dados
        tsv_file (str): Caminho para o arquivo TSV contendo as séries temporais
        K (int): Número de vizinhos mais próximos para construir o grafo
        alpha (float): Parâmetro de decaimento para os pesos das arestas

    Atributos:
        root (str): Diretório raiz dos dados
        data_type (str): Tipo dos dados ("processed")
        save_path (str): Caminho para salvar os dados processados
        K (int): Número de vizinhos mais próximos
        alpha (float): Parâmetro de decaimento
        graph (dgl.DGLGraph): Grafo DGL processado
        labels (torch.Tensor): Rótulos das classes

    Propriedades:
        save_path (str): Caminho para salvar dados processados
        data_type (str): Tipo dos dados
        labels (torch.Tensor): Rótulos das classes
        num_classes (int): Número de classes únicas

    Métodos:
        process(): Processa os dados brutos e gera o grafo
        save(): Salva os dados processados
        load(): Carrega dados processados
        has_cache(): Verifica existência de dados processados
        plot_graph(): Visualiza o grafo gerado
    """
    def __init__(self, root, tsv_file, K, alpha):
        print("Creating dataset...")
        self.root = root
        self.data_type = "processed"
        
        graph_info = f"K_{K}_alpha_{str(alpha).replace('.', '')}"
        
        self.save_path = osp.join(self.root, graph_info, self.data_type)

        with open(tsv_file, "r") as file:
            self.__train_data = file.readlines()
        
        labels = np.array([int(line.split("\t")[0]) for line in self.__train_data])
        self.labels = torch.tensor(np.array([sorted(list(set(labels))).index(l) for l in labels]))
        
        self.num_classes = len(np.unique(self.labels))        

        self.K = K
        self.alpha = alpha

        super().__init__(
            name="SimTSCDataset",
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

    def process(self):
        """
        Processa os dados brutos e cria um grafo DGL.
        
        Este método:
        1. Converte as séries temporais em uma matriz de distâncias DTW
        2. Para cada série temporal:
            - Seleciona os K vizinhos mais próximos
            - Calcula os pesos das arestas usando uma função exponencial negativa: 1/exp(alpha * distância)
            - Normaliza os pesos das arestas para criar uma distribuição de probabilidade (soma = 1)
        3. Constrói um grafo DGL onde:
            - Cada nó representa uma série temporal
            - As arestas conectam séries temporais similares
            - Os pesos das arestas formam uma distribuição de probabilidade que indica:
                * Séries mais similares (menor distância) recebem maior probabilidade
                * Séries menos similares (maior distância) recebem menor probabilidade
                * A soma das probabilidades das arestas de saída de cada nó é 1
                * O parâmetro alpha controla o quão acentuada é a distribuição
            - Os rótulos são armazenados como atributos dos nós
        
        O grafo resultante captura a estrutura de similaridade entre as séries temporais,
        permitindo análises baseadas em grafos e aprendizado de representações.
        """
        time_series = np.array([np.array(line.split("\t")[1:], dtype=np.float32) for line in self.__train_data])
        distances = distance_matrix.create_dtw_matrix(time_series)
        adj = torch.from_numpy(distances.astype(np.float32))
        ranks = torch.argsort(adj, dim=1)
        sparse_index = [[], []]
        edge_weights = []
        
        for i in tqdm(range(len(adj)), desc='Calculating edge weights and sampling edges'):
            _sparse_value = []
            for j in ranks[i][:self.K]:
                sparse_index[0].append(i)
                sparse_index[1].append(j)
                _sparse_value.append(1/np.exp(self.alpha*adj[i][j]))
            _sparse_value = np.array(_sparse_value)
            _sparse_value /= _sparse_value.sum()
            edge_weights.extend(_sparse_value.tolist())
        
        # sparse_index = torch.LongTensor(sparse_index)
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1)        
        src, dst = sparse_index[0], sparse_index[1]
        
        dgl_graph = dgl.graph((src, dst), num_nodes=len(adj))
        dgl_graph.edata["weight"] = edge_weights
        dgl_graph.ndata['label'] = self.labels
        
        self.graph = dgl_graph
    def __getitem__(self, idx):
        return self.graph[idx]

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
                    "K" : self.K,
                    "alpha" : self.alpha
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
            self.K = load_info(info_path)["K"]
            self.alpha = load_info(info_path)["alpha"]
            
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
        nx_g = self.graph.to_networkx().to_undirected()
        pos = nx.spring_layout(nx_g)  # Layout for visualization
        plt.figure(figsize=(8, 8))
        nx.draw(nx_g, pos, with_labels=True, node_size=500, node_color="skyblue", alpha=0.7, edge_color="gray")
        plt.show()

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_file):        
        with open(tsv_file, "r") as file:
            self.data = file.readlines()            
        
        self.time_series = np.array([np.array(d.split("\t")[1:], dtype=np.float32) for d in self.data])
        
        labels = np.array([int(line.split("\t")[0]) for line in self.data])
        self.labels = torch.tensor(np.array([sorted(list(set(labels))).index(l) for l in labels]))
    
    @property
    def num_classes(self):
        return len(np.unique(self.labels))

    def __getitem__(self, idx):
        return self.time_series[idx], self.labels[idx]

    def __len__(self):
        return len(self.time_series)