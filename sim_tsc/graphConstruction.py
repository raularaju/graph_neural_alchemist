
import torch
import numpy as np
import dgl
import networkx as nx
from .distance_matrix import create_dtw_matrix, create_pearson_matrix

import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dist_matrix(time_series, type_step, batch_idx, dataset_path, strategy='dtw'):
    if(len(np.shape(time_series)) <= 1):
        raise ValueError('The time_series set must be a 2D array with more than one time series')
    save_path = os.path.join(f"{dataset_path}/{strategy}_matrix_{type_step}_{batch_idx}.npy")
    if os.path.exists(save_path):
        return np.load(save_path)    
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    if strategy == 'dtw':
        return create_dtw_matrix(time_series, save_path)
    elif strategy == 'pearson':
        return create_pearson_matrix(time_series, save_path)
    else:
        raise ValueError('Invalid strategy')

def construct_graph_dtw(time_series, labels, K, alpha, type_step, batch_idx, dataset_path, adj_only=False):
    """
    Constrói um grafo usando a distância DTW (Dynamic Time Warping) entre séries temporais.
    
    Este método:
    1. Calcula a matriz de distâncias DTW entre as séries temporais
    2. Seleciona os K vizinhos mais próximos para cada série
    3. Calcula os pesos das arestas usando uma função exponencial negativa
    4. Constrói um grafo DGL ou uma matriz de adjacência esparsa
    
    Parâmetros
    ----------
    time_series : torch.Tensor
        Tensor contendo as séries temporais
    labels : torch.Tensor
        Tensor contendo os rótulos das séries
    K : int
        Número de vizinhos mais próximos a considerar
    alpha : float
        Parâmetro de escala para o cálculo dos pesos das arestas
    type_step : str
        Identificador do passo (treino/validação/teste)
    batch_idx : int
        Índice do lote atual
    dataset_path : str
        Caminho para salvar/carregar a matriz de distâncias
    adj_only : bool, opcional
        Se True, retorna apenas a matriz de adjacência. Padrão: False
        
    Retorna
    -------
    Union[dgl.DGLGraph, torch.sparse.FloatTensor]
        Grafo DGL ou matriz de adjacência esparsa, dependendo do parâmetro adj_only
    """
    time_series = time_series.cpu().numpy()
    labels = labels.cpu().numpy()
    
    distances = get_dist_matrix(time_series, type_step, batch_idx, dataset_path, strategy='dtw')    
    adj_matrix = torch.from_numpy(distances.astype(np.float32))    
    ranks = torch.argsort(adj_matrix, dim=1)
    sparse_index = [[], []]
    edge_weights = []    
    for i in range(len(adj_matrix)):
        _sparse_value = []
        for j in ranks[i][:K]:
            sparse_index[0].append(i)
            sparse_index[1].append(j)
            _sparse_value.append(1/np.exp( np.float128(alpha*adj_matrix[i][j])) )
        _sparse_value = np.array(_sparse_value)
        _sparse_value /= _sparse_value.sum()
        edge_weights.extend(_sparse_value.tolist())
    
    if adj_only:
        sparse_index = torch.LongTensor(sparse_index)        
        edge_weights = torch.FloatTensor(edge_weights)
        adj_matrix = torch.sparse.FloatTensor(sparse_index, edge_weights, adj_matrix.size())
        return adj_matrix.to(device)
    
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1)
    src, dst = sparse_index[0], sparse_index[1]

    #create a a weighted graph using sparse_index and sparse_value
    dgl_graph = dgl.graph((src, dst), num_nodes=len(adj_matrix))
    dgl_graph.edata["weight"] = edge_weights
    dgl_graph.ndata['label'] = torch.tensor(labels, dtype=torch.long).view(-1, 1)
        
    return dgl_graph.to(device)

def construct_graph_pearson(time_series, labels, R, type_step, batch_idx, dataset_path, adj_only=False):    
    time_series = time_series.cpu().numpy()
    labels = labels.cpu().numpy()
    
    distances = get_dist_matrix(time_series, type_step, batch_idx, dataset_path, strategy='pearson')
    adj_matrix = np.absolute(distances)
    
    #if the correlation is below the threshold R, remove the edge, otherwise keep the edge
    adj_matrix[adj_matrix < R] = 0
    adj_matrix[adj_matrix >= R] = 1
    
    if adj_only:
        adj_matrix = torch.from_numpy(adj_matrix.astype(np.float32))
        return adj_matrix.to(device)
    
    dgl_graph = dgl.from_networkx(nx.from_numpy_array(adj_matrix))  
    dgl_graph.ndata['feat'] = torch.tensor(time_series, dtype=torch.float32)
    dgl_graph.ndata['label'] = torch.tensor(labels, dtype=torch.long).view(-1, 1)    
        
    return dgl_graph.to(device)