import torch
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import dgl
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info
import traceback

from time2graph.run_time2graph import run_time2graph
import networkx as nx

class Time2GraphDataset(dgl.data.DGLDataset):
    """
    Dataset personalizado para manipular dados de séries temporais convertidos em grafos usando o método Time2Graph.

    Esta classe implementa um dataset que:
    1. Recebe séries temporais como entrada
    2. Extrai shapelets sensíveis ao tempo usando o método Time2Graph
    3. Constrói representações em grafo das séries temporais
    4. Armazena os grafos resultantes no formato DGL

    Args:
        root (str): Diretório raiz para armazenar os dados
        tsv_file (str): Caminho para o arquivo TSV contendo as séries temporais
        args (argparse.Namespace): Argumentos de configuração para o Time2Graph

    Atributos:
        root (str): Diretório raiz dos dados
        data_type (str): Tipo dos dados ("processed")
        save_path (str): Caminho para salvar os dados processados
        args: Argumentos de configuração
        tsv_path (str): Caminho do arquivo de entrada
        graphs (list): Lista de grafos DGL processados
        classes (list): Lista de rótulos das classes

    Propriedades:
        save_path (str): Caminho para salvar dados processados
        data_type (str): Tipo dos dados
        num_features (int): Número de features dos nós
        labels (array): Rótulos das classes
        num_classes (int): Número de classes únicas

    Métodos:
        process(): Processa os dados brutos e gera os grafos
        save(): Salva os dados processados
        load(): Carrega dados processados
        has_cache(): Verifica existência de dados processados
    """
    def __init__(self, root, tsv_file, args):
        print("Creating dataset...")
        self.root = root
        self.data_type = "processed"
        self.save_path = osp.join(self.root, self.data_type)        
        self.args = args
        self.tsv_path = tsv_file
        
        self.graphs = []
        self.classes = []

        super().__init__(
            name="Time2GraphDataset",
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
        1. Gera shapelets sensíveis ao tempo e aprende representações em grafo usando Time2Graph
        2. Para cada matriz de adjacência:
            - Converte para formato DGL
            - Adiciona features dos nós
            - Armazena o grafo e seu rótulo
        3. Atualiza metadados do dataset (número de features e classes)
        
        Os grafos resultantes contêm:
        - Nós representando shapelets extraídos da série temporal
        - Arestas baseadas nas relações temporais entre shapelets
        - Features dos nós aprendidas pelo Time2Graph
        """
        print("Generating time-aware shapelets and learning graph representations...")        
        self.args.tsv_file = self.tsv_path
        self.args.root_path = self.root
        node_features, adj_matrix, labels = run_time2graph(
            self.args
        )
        
        # print(f"shape of node_features: {node_features.shape}")

        for i, matrix in tqdm(enumerate(adj_matrix), desc="Processing dataset"):
            dgl_graph = dgl.from_networkx(nx.from_numpy_array(matrix))    
            dgl_graph.ndata["feat"] = torch.tensor(node_features[i], dtype=torch.float32)
            y = torch.tensor(labels[i], dtype=torch.long)
            self.graphs.append(dgl_graph)
            self.classes.append(y)
        
        self.classes = torch.LongTensor(self.classes)
        self.num_features = node_features.shape[-1]
        self.num_classes = len(np.unique(labels))
    def __getitem__(self, idx):
        return self.graphs[idx], self.classes[idx]
    
    def __len__(self):
        return len(self.graphs)
    
    def save(self):
        print(f"Saving processed data in {self.save_path}")
        graph_path = os.path.join(self.save_path, "_dgl_graph.bin")
        try:
            save_graphs(graph_path, self.graphs, {"labels": self.classes})
            info_path = os.path.join(self.save_path, "_info.pkl")
            save_info(
                info_path,
                {
                    "num_classes": self.num_classes,
                    "num_features": self.num_features,
                    "args": self.args,
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
            self.graphs, label_dict = load_graphs(graph_path)
            self.classes = label_dict["labels"]
            info_path = os.path.join(self.save_path, "_info.pkl")
            self.num_classes = load_info(info_path)["num_classes"]
            self.num_features = load_info(info_path)["num_features"]
            self.args = load_info(info_path)["args"]
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