from os import makedirs
import networkx as nx
import random
import numpy as np
from typing import List
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
import dgl
from datasets import Time2GraphDataset
import xgboost as xgb
from sklearn.metrics import classification_report
from .utils.base_utils import Debugger

class DeepWalk:
    """
    Implementa o algoritmo DeepWalk para gerar embeddings de nós em um grafo.

    O DeepWalk utiliza passeios aleatórios para gerar sequências de nós, que são então
    usadas para treinar um modelo Word2Vec e gerar representações vetoriais (embeddings)
    para cada nó do grafo.

    O processo consiste em:
    1. Gerar múltiplos passeios aleatórios começando de cada nó
    2. Usar esses passeios como "sentenças" para treinar um modelo Word2Vec
    3. Extrair os embeddings resultantes para cada nó

    Os embeddings capturam características estruturais do grafo e podem ser usados
    para tarefas como classificação de nós e detecção de comunidades.
    """
    def __init__(self, window_size: int, embedding_size: int, walk_length: int, walks_per_node: int):
        """
        :param window_size: tamanho da janela para o modelo Word2Vec
        :param embedding_size: tamanho do embedding final
        :param walk_length: comprimento do passeio aleatório
        :param walks_per_node: número de passeios por nó
        """
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.walk_length = walk_length
        self.walk_per_node = walks_per_node

    def random_walk(self, g: nx.Graph, start: str, use_probabilities: bool = False) -> List[str]:
        """
        Gera um passeio aleatório começando no nó inicial
        
        :param g: Grafo
        :param start: nó inicial para o passeio aleatório
        :param use_probabilities: se True, considera os pesos atribuídos a cada aresta para selecionar o próximo candidato
        :return: Lista de nós visitados durante o passeio
        """
        walk = [start]
        for i in range(self.walk_length):
            neighbours = g.neighbors(walk[i])
            neighs = list(neighbours)
            if len(neighs) == 0:
                break
            if use_probabilities:
                probabilities = [g.get_edge_data(walk[i], neig)["weight"] for neig in neighs]
                sum_probabilities = sum(probabilities)
                probabilities = list(map(lambda t: t / sum_probabilities, probabilities))
                p = np.random.choice(neighs, p=probabilities)
            else:
                p = random.choice(neighs)
            walk.append(p)
        return walk

    def get_walks(self, g: nx.Graph, use_probabilities: bool = False) -> List[List[str]]:
        """
        Gera todos os passeios aleatórios
        
        :param g: Grafo
        :param use_probabilities: se True, usa probabilidades baseadas nos pesos das arestas
        :return: Lista de passeios aleatórios
        """
        random_walks = []
        for _ in range(self.walk_per_node):
            random_nodes = list(g.nodes)
            random.shuffle(random_nodes)
            for node in random_nodes:
                random_walks.append(self.random_walk(g=g, start=node, use_probabilities=use_probabilities))
        return random_walks

    def compute_embeddings(self, walks: List[List[str]]):
        """
        Calcula os embeddings dos nós para os passeios gerados
        
        :param walks: Lista de passeios
        :return: Modelo Word2Vec treinado com os embeddings dos nós
        """
        model = Word2Vec(sentences=walks, window=self.window_size, vector_size=self.embedding_size)
        return model.wv

def get_graph_embedding(graph: nx.Graph, model: DeepWalk) -> np.ndarray:
    """
    Compute the graph embedding using the DeepWalk model
    :param graph: Graph
    :param model: DeepWalk model
    :return:
    """
    walks = model.get_walks(graph)
    node_embeddings_dic = model.compute_embeddings(walks)
    node_embeddings = node_embeddings_dic.get_normed_vectors()
    graph_embedding = np.mean(node_embeddings, axis=0)
    return graph_embedding

def get_graph_embeddings(dataset: Time2GraphDataset, model: DeepWalk) -> np.ndarray:
    """
    Compute the graph embeddings for the dataset
    :param dataset: Time2GraphDataset
    :param model: DeepWalk model
    :return:
    """
    graph_embeddings = []
    for dgl_graph in tqdm(dataset.graphs):
        graph = dgl.to_networkx(dgl_graph)
        graph_embeddings.append(get_graph_embedding(graph, model))
    return np.array(graph_embeddings)


def run_xgboost(dataset_train : Time2GraphDataset, dataset_test : Time2GraphDataset, args):
    """
    Executa o modelo XGBoost para classificação usando embeddings de grafos.
    
    Esta função:
    1. Gera embeddings dos grafos usando DeepWalk
    2. Treina um classificador XGBoost nos embeddings
    3. Avalia o modelo no conjunto de teste
    4. Salva as predições e métricas de avaliação
    
    Parâmetros:
        dataset_train: Time2GraphDataset
            Dataset de treino contendo os grafos e rótulos
        dataset_test: Time2GraphDataset 
            Dataset de teste contendo os grafos e rótulos
        args: objeto
            Argumentos de configuração contendo:
            - save_dir: Diretório para salvar resultados
            - dataset: Nome do dataset
            
    Retorna:
        None
    """
    model = DeepWalk(window_size=2, embedding_size=100, walk_length=15, walks_per_node=100)
    train_graph_embeddings = get_graph_embeddings(dataset_train, model)
    test_graph_embeddings = get_graph_embeddings(dataset_test, model)
    train_labels = dataset_train.classes
    test_labels = dataset_test.classes
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    # Fit the model on the training data
    model.fit(train_graph_embeddings, train_labels)

    # Predict on the test set
    y_pred = model.predict(test_graph_embeddings)

    # Print the classification report (includes precision, recall, f1-score, accuracy)
    print(classification_report(test_labels, y_pred))
    
    makedirs(args.save_dir, exist_ok=True)        
    #store the y_preds on the results path
    np.save(f"{args.save_dir}/{args.dataset}_y_pred.npy", y_pred)
    
    report = classification_report(y_true=test_labels, y_pred=y_pred, output_dict=True)
    
    Debugger.info_print(f"Classification report: {report}")

    
    #save the values for classification report and mcc into a formatted .csv file
    with open(f"{args.save_dir}/{args.dataset}_classification_report.csv", "w") as f:
        print(f"saving classification report to {args.save_dir}/{args.dataset}_classification_report.csv")
        f.write("metric, value\n")
        for key, value in report.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    f.write(f"{key}_{k}, {v}\n")
            else:
                f.write(f"{key}, {value}\n") 

    
