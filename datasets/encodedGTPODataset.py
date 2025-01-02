import torch
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import dgl
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info
import traceback
import ordpy

class encodedGTPODataset(dgl.data.DGLDataset):
    """
    Dataset para grafos codificados GTPO (Graph Time Pattern Ordinal).
    
    Esta classe herda de DGLDataset e implementa um dataset personalizado para trabalhar com 
    padrões ordinais codificados em grafos. O dataset processa séries temporais em grafos 
    usando redes ordinais e codifica os padrões como características dos nós.

    Parâmetros
    ----------
    root : str
        Caminho raiz onde os dados serão salvos
    tsv_file : str
        Arquivo TSV contendo os dados de entrada
    encoder : sklearn.preprocessing.OneHotEncoder
        Codificador OneHotEncoder já treinado para transformar os padrões ordinais em características dos nós.
        Deve ser treinado previamente com os padrões ordinais do conjunto de dados.
    num_features : int, opcional (padrão=2)
        Número inicial de características dos nós
    op_length : int, opcional (padrão=3)
        Comprimento do padrão ordinal

    Atributos
    ----------
    root : str
        Diretório raiz do dataset
    data_type : str
        Tipo de dados (processado)
    save_path : str
        Caminho onde os dados processados são salvos
    labels : numpy.ndarray
        Rótulos das classes
    num_classes : int
        Número de classes únicas
    graph : list
        Lista de grafos DGL
    classes : list
        Lista de rótulos correspondentes aos grafos
    encoder : sklearn.preprocessing.OneHotEncoder
        Codificador OneHotEncoder usado para transformar os padrões ordinais
    """
    def __init__(self, root, tsv_file, encoder, num_features=2, op_length=3):
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
        self.encoder = encoder

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
    def encoder(self):
        return self.__encoder

    @encoder.setter
    def encoder(self, encoder):
        self.__encoder = encoder

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
        Processa os dados brutos e cria grafos DGL com características codificadas.
        
        Este método:
        1. Itera sobre os dados de treinamento
        2. Extrai rótulos e sinais das séries temporais
        3. Cria redes ordinais usando ordpy
        4. Converte as redes em grafos DGL
        5. Codifica os padrões ordinais como características dos nós usando one-hot encoding
        6. Armazena os grafos e classes processados
        
        Os grafos resultantes contêm:
        - Nós representando padrões ordinais
        - Arestas ponderadas entre padrões consecutivos
        - Características dos nós codificadas em one-hot
        - Rótulos das classes como tensores
        """
        for data in tqdm(self.__train_data, desc="Processing dataset"):
            label = int(data.split("\t")[0])
            label = sorted(list(set(self.labels))).index(label)
            signal = np.array(data.split("\t")[1:]).astype(np.float32)

            nodes, edges, edge_weights = ordpy.ordinal_network(signal, self.op_length)

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

            # Encode ordinal patterns as node features using one-hot encoding
            encoded_patterns = self.encoder.transform(np.array(nodes).reshape(-1, 1))

            dgl_graph.ndata["feat"] = torch.tensor(
                encoded_patterns, dtype=torch.float32
            )

            y = torch.tensor(label, dtype=torch.long)

            # Add self loop for graph convolutions
            # dgl_graph = dgl_graph.add_self_loop()

            self.graph.append(dgl_graph)
            self.classes.append(y)

        self.classes = torch.LongTensor(self.classes)
        self.num_features = encoded_patterns.shape[1]
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
                    "encoder": self.encoder,
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
            self.encoder = load_info(info_path)["encoder"]
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