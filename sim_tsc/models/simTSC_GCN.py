import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import lightning as pl
from .ResNetBlock import ResNetBlock

'''
    Implementação do SimTSC usando Redes Neurais Convolucionais em Grafos e Blocos ResNet, similar a https://arxiv.org/abs/2201.01413
'''
    
class GraphConvolution(pl.LightningModule):
    """
    Camada GCN simples, similar à implementada em https://arxiv.org/abs/1609.02907
    
    Esta classe implementa uma camada de convolução em grafos (GCN) que realiza:
    - Transformação linear dos recursos de entrada
    - Propagação da mensagem através da matriz de adjacência
    - Adição opcional de viés (bias)
    
    Parâmetros
    ----------
    in_features : int
        Número de características de entrada
    out_features : int 
        Número de características de saída
    bias : bool, opcional
        Se True, adiciona um termo de viés. Padrão: True
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_matrix):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj_matrix, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
class simTSC_GCN(pl.LightningModule):
    """
    Implementa o modelo SimTSC usando GCN (Graph Convolutional Network).
    
    Esta classe implementa uma rede neural que combina:
    - 3 blocos ResNet para processamento de séries temporais
    - 3 camadas de convolução em grafos (GCN) para aprendizado das relações entre amostras
    
    Parâmetros
    ----------
    args : objeto
        Objeto contendo os argumentos de configuração:
        - nhid: Número de mapas de características nas camadas intermediárias
        - num_classes: Número de classes para classificação
        - n_channels: Número de canais de entrada da série temporal
        - dropout: Taxa de dropout aplicada entre as camadas GCN
        - seed: Semente aleatória para reprodutibilidade
    """
    def __init__(self, args):
        self.n_feature_maps = args.nhid
        self.num_classes = args.num_classes
        self.n_channels = args.n_channels
        self.dropout = args.dropout
        
        self.seed = args.seed        
        
        super(simTSC_GCN, self).__init__()
        self.layer1 = ResNetBlock(self.n_channels, self.n_feature_maps)
        self.layer2 = ResNetBlock(self.n_feature_maps, self.n_feature_maps)
        self.layer3 = ResNetBlock(self.n_feature_maps, self.n_feature_maps)        
        
        self.fc = nn.Linear(self.n_feature_maps, self.num_classes)
        
        self.gc1 = GraphConvolution(self.n_feature_maps, self.n_feature_maps)
        self.gc2 = GraphConvolution(self.n_feature_maps, self.n_feature_maps)
        self.gc3 = GraphConvolution(self.n_feature_maps, self.num_classes)
        
    def forward(self, x, adj_matrix):
        """
        Propaga os dados através da rede.
        
        Parâmetros
        ----------
        x : torch.Tensor
            Tensor de entrada contendo as séries temporais
        adj_matrix : torch.Tensor
            Matriz de adjacência do grafo
            
        Retorna
        -------
        torch.Tensor
            Log-probabilidades das classes preditas
        """
        if self.n_channels == 1:
            x = x.unsqueeze(1) # Garante que x tem forma (batch_size, 1, seq_len) quando a série temporal é univariada
            
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)        
        x = F.avg_pool1d(x, x.shape[-1]).squeeze()
        # x = x.view(x.size(0), -1)
        
        h = F.relu(self.gc1(x, adj_matrix))
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(self.gc2(h, adj_matrix))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.gc3(h, adj_matrix)
        
        return F.log_softmax(h, dim=1)        
