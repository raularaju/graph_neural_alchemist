import torch
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import dgl
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info
import traceback
from utils import compute_pagerank
import ordpy

'''
    Grafo de Transição de Padrões Ordinais baseado no método de Bandt-Pompe para extração de Padrões Ordinais
    
'''

class GTPODataset(dgl.data.DGLDataset):
    """
    Dataset personalizado para manipular dados de séries temporais convertidos em grafos usando Padrões Ordinais.

    Esta classe implementa um dataset que:
    1. Recebe séries temporais como entrada
    2. Extrai padrões ordinais usando o método de Bandt-Pompe
    3. Constrói um grafo de transição entre os padrões
    4. Calcula features dos nós usando PageRank e grau dos nós
    5. Armazena o grafo resultante no formato DGL

    Args:
        root (str): Diretório raiz para armazenar os dados
        tsv_file (str): Caminho para o arquivo TSV contendo as séries temporais
        num_features (int): Número de features dos nós (padrão: 2)
        op_length (int): Comprimento dos padrões ordinais (padrão: 3)

    Atributos:
        root (str): Diretório raiz dos dados
        data_type (str): Tipo dos dados ("processed")
        save_path (str): Caminho para salvar os dados processados
        op_length (int): Comprimento dos padrões ordinais
        graph (list): Lista de grafos DGL processados
        classes (torch.Tensor): Rótulos das classes
        num_features (int): Número de features dos nós
        num_classes (int): Número de classes únicas

    Propriedades:
        op_length (int): Comprimento dos padrões ordinais
        save_path (str): Caminho para salvar dados processados
        data_type (str): Tipo dos dados
        num_features (int): Número de features dos nós
        labels (torch.Tensor): Rótulos das classes
        num_classes (int): Número de classes únicas

    Métodos:
        process(): Processa os dados brutos e gera os grafos
        save(): Salva os dados processados
        load(): Carrega dados processados
        has_cache(): Verifica existência de dados processados
        is_directed(): Verifica se os grafos são direcionados
    """
    def __init__(self, root, tsv_file, num_features=2, op_length=3):
        print("Creating dataset...")
        self.root = root
        self.data_type = "processed"
        self.save_path = osp.join(self.root, self.data_type)

        with open(tsv_file, "r") as file:
            self.__train_data = file.readlines()

        self.labels = np.array([int(line.split("\t")[0]) for line in self.__train_data])
        print(f"unique labels: {list(set(self.labels))}")
        self._num_features = num_features
        self.num_classes = len(np.unique(self.labels))
        self.graph = []
        self.classes = []

        self.op_length = op_length

        super().__init__(
            name="GTPODataset",
            raw_dir=root,
            save_dir=self.save_path,
            force_reload=False,
            verbose=False,
        )
        print("Dataset created!")

    def download(self):
        pass

    @property
    def op_length(self):
        return self.__op_length

    @op_length.setter
    def op_length(self, op_length):
        self.__op_length = op_length

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
    def num_features(self):
        return self._num_features

    @num_features.setter
    def num_features(self, value):
        self._num_features = value

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
        Processa os dados brutos e cria grafos DGL.
        
        Este método:
        1. Para cada série temporal no conjunto de dados:
            - Extrai o rótulo e o sinal da série
            - Gera uma rede ordinal usando ordpy.ordinal_network que:
                * Mapeia padrões ordinais para nós
                * Cria arestas direcionadas entre padrões consecutivos
                * Normaliza os pesos das arestas
            - Converte a rede ordinal para um grafo DGL onde:
                * Cada nó representa um padrão ordinal
                * As arestas representam transições entre padrões
                * Os pesos das arestas são normalizados
            - Calcula features dos nós:
                * PageRank: centralidade baseada em caminhadas aleatórias
                * Grau do nó: número de conexões de entrada
            - Armazena o grafo e seu rótulo
        2. Atualiza metadados do dataset (número de features)
        
        O grafo resultante captura a dinâmica da série temporal através:
            - Da topologia da rede ordinal (estrutura de transições)
            - Dos pesos das arestas (frequência das transições)
            - Das features dos nós (importância estrutural dos padrões)
        """
        for data in tqdm(self.__train_data, desc="Processing dataset"):
            label = int(data.split("\t")[0])
            label = sorted(list(set(self.labels))).index(label)
            signal = np.array(data.split("\t")[1:]).astype(np.float32)

            nodes, edges, edge_weights = ordpy.ordinal_network(
                data=signal,
                dx=self.op_length,
                normalized=True,
                overlapping=True,
                directed=True,                
            )

            # Create a mapping from node labels to node indices
            node_to_index = {node: idx for idx, node in enumerate(nodes)}

            # Extract source and destination indices from edges
            src = [node_to_index[edge[0]] for edge in edges]
            dst = [node_to_index[edge[1]] for edge in edges]

            # Create a DGL graph from the src and dst arrays
            dgl_graph = dgl.graph((src, dst), num_nodes=len(nodes))
            dgl_graph.edata["weight"] = torch.tensor(
                edge_weights, dtype=torch.float32
            ).view(-1, 1)
            
            
            # Calulando as features de cada nó: PageRank, Grau do Nó            
            
            N, K = dgl_graph.number_of_nodes(), 100
            page_ranks = compute_pagerank(dgl_graph, N=N, K=K)
            node_degree = dgl_graph.in_degrees(dgl_graph.nodes()).type(torch.float32)            

            x = torch.stack([page_ranks, node_degree], dim=1)
            
            dgl_graph.ndata["feat"] = x            
            y = torch.tensor(label, dtype=torch.long)

            # Add self loop for graph convolutions
            # dgl_graph = dgl_graph.add_self_loop()

            self.graph.append(dgl_graph)
            self.classes.append(y)

        self.classes = torch.LongTensor(self.classes)
        self.num_features = x.shape[1]
    def __getitem__(self, idx):
        return self.graph[idx], self.classes[idx]

    def __len__(self):
        return len(self.graph)

    def save(self):
        print(f"Saving processed data in {self.save_path}")
        graph_path = os.path.join(self.save_path, "_dgl_graph.bin")
        try:
            save_graphs(graph_path, self.graph, {"labels": self.classes})
            info_path = os.path.join(self.save_path, "_info.pkl")
            save_info(
                info_path,
                {
                    "num_classes": self.num_classes,
                    "num_features": self.num_features,
                    "op_length": self.op_length,
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
            self.graph, label_dict = load_graphs(graph_path)
            self.classes = label_dict["labels"]
            info_path = os.path.join(self.save_path, "_info.pkl")
            self.num_classes = load_info(info_path)["num_classes"]
            self.num_features = load_info(info_path)["num_features"]
            self.op_length = load_info(info_path)["op_length"]
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

    def is_directed(self):
        for graph in self.graph:
            if not dgl.to_networkx(graph).is_directed():
                return False
        return True