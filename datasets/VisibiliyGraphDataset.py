import torch
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from ts2vg import NaturalVG
from utils import compute_pagerank
import dgl
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info
import traceback

class VisibiliyGraphDataset(dgl.data.DGLDataset):
    """
    VGDatasetDGL é uma classe personalizada de dataset para manipular datasets de grafos de visibilidade no formato DGL.

    Args:
        root (str): Diretório raiz do dataset.
        tsv_file (str): Caminho para o arquivo TSV contendo o dataset, no formato [rótulo \t sinal].
        directed (str, opcional): Direção das arestas do grafo de visibilidade. Padrão é "left_to_right". Use None para grafos não direcionados.
        weighted (str, opcional): Esquema de pesos para as arestas do grafo de visibilidade. Padrão é "sq_distance".
        edge_attrs (list, opcional): Lista de atributos das arestas. Padrão é ["weight"].
        num_features (int, opcional): Número de características para cada nó. Padrão é 2.

    Atributos:
        root (str): Diretório raiz do dataset.
        data_type (str): Tipo do dataset.
        save_path (str): Caminho para salvar os dados processados.
        labels (numpy.ndarray): Array de rótulos para cada amostra de dados.
        num_classes (int): Número de classes únicas no dataset.
        graphs (list): Lista de grafos DGL.
        classes (list): Lista de rótulos de classe para cada grafo.

    Propriedades:
        directed (str): Direção das arestas do grafo de visibilidade.
        weighted (str): Esquema de pesos para as arestas do grafo de visibilidade.
        edge_attrs (list): Lista de atributos das arestas.
        save_path (str): Caminho para salvar os dados processados.
        data_type (str): Tipo do dataset.
        num_features (int): Número de características para cada nó.
        labels (numpy.ndarray): Array de rótulos para cada amostra de dados.
        num_classes (int): Número de classes únicas no dataset.

    Métodos:
        download(): Método placeholder para download do dataset.
        process(): Processa o dataset e cria grafos DGL.
        __getitem__(idx): Obtém o grafo e seu rótulo de classe correspondente no índice fornecido.
        __len__(): Obtém o número de grafos no dataset.
        save(): Salva os dados processados.
        load(): Carrega os dados processados.
        has_cache(): Verifica se existem dados processados.
        is_directed(): Verifica se todos os grafos no dataset são direcionados.
    """

    def __init__(
        self,
        root,
        tsv_file,
        directed="left_to_right",
        weighted="sq_distance",
        edge_attrs=["weight"],
        num_features=3,
    ):
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

        self.directed = directed
        self.weighted = weighted
        self.edge_attrs = edge_attrs

        super().__init__(
            name="VGDatasetDGL",
            raw_dir=root,
            save_dir=self.save_path,
            force_reload=False,
            verbose=False,
        )
        print("Dataset created!")

    def download(self):
        pass

    @property
    def directed(self):
        return self.__directed

    @directed.setter
    def directed(self, directed):
        if directed == None:
            self.__directed = None
        else:
            self.__directed = directed

    @property
    def weighted(self):
        return self.__weighted

    @weighted.setter
    def weighted(self, weighted):
        if weighted == None:
            self.__weighted = None
        else:
            self.__weighted = weighted

    @property
    def edge_attrs(self):
        return self.__edge_attrs

    @edge_attrs.setter
    def edge_attrs(self, edge_attrs):
        if edge_attrs == None:
            self.__edge_attrs = None
        else:
            self.__edge_attrs = edge_attrs

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
        1. Itera sobre os dados de treinamento
        2. Para cada série temporal:
            - Extrai o rótulo e o sinal
            - Constrói o grafo de visibilidade natural
            - Converte para formato DGL
            - Calcula features dos nós (PageRank, grau e valor do sinal)
            - Adiciona as features ao grafo
        3. Armazena os grafos e rótulos processados
        
        Os grafos resultantes contêm:
        - Nós representando pontos da série temporal
        - Arestas baseadas na visibilidade entre pontos
        - Features dos nós: PageRank, grau do nó e valor do sinal
        """
        for data in tqdm(self.__train_data, desc="Processing dataset"):
            label = int(data.split("\t")[0])
            label = sorted(list(set(self.labels))).index(label)
            signal = np.array(data.split("\t")[1:]).astype(np.float32)

            visibility_graph = (
                NaturalVG(directed=self.directed, weighted=self.weighted)
                .build(signal)
                .as_networkx()
            )

            dgl_graph = dgl.from_networkx(visibility_graph, edge_attrs=self.edge_attrs)

            N, K = dgl_graph.number_of_nodes(), 10000

            # calculando as features de cada nó
            page_ranks = compute_pagerank(dgl_graph, N=N, K=K)
            node_degree = dgl_graph.in_degrees(dgl_graph.nodes()).type(torch.float32)
            signal = torch.tensor(signal, dtype=torch.float32)

            x = torch.stack([page_ranks, node_degree, signal], dim=1)
            y = torch.tensor(label, dtype=torch.long)

            dgl_graph.ndata["feat"] = x

            self.graph.append(dgl_graph)
            self.classes.append(y)

        self.classes = torch.LongTensor(self.classes)
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
                {"num_classes": self.num_classes, "num_features": self.num_features},
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