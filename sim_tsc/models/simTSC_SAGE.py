import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
import lightning as pl
from .ResNetBlock import ResNetBlock

'''
    Implementação do SimTSC usando SAGEConv do DGL e Blocos ResNet, adaptado de https://arxiv.org/abs/2201.01413
'''

class simTSC_SAGE(pl.LightningModule):
    """
    Implementa o modelo SimTSC usando GraphSAGE.
    
    Esta classe implementa uma rede neural que combina:
    - 3 blocos ResNet para processamento de séries temporais
    - 3 camadas de convolução GraphSAGE para aprendizado das relações entre amostras
    
    Parâmetros
    ----------
    args : objeto
        Objeto contendo os argumentos de configuração:
        - nhid: Número de mapas de características nas camadas intermediárias
        - num_classes: Número de classes para classificação
        - n_channels: Número de canais de entrada da série temporal
        - agg_type: Tipo de agregação usado no GraphSAGE
        - seed: Semente aleatória para reprodutibilidade
    """
    def __init__(self, args):
        self.n_feature_maps = args.nhid
        self.num_classes = args.num_classes
        self.n_channels = args.n_channels
        self.agg_type = args.agg_type
        self.seed = args.seed
        
        super(simTSC_SAGE, self).__init__()
        self.layer1 = ResNetBlock(self.n_channels, self.n_feature_maps)
        self.layer2 = ResNetBlock(self.n_feature_maps, self.n_feature_maps)
        self.layer3 = ResNetBlock(self.n_feature_maps, self.n_feature_maps)        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.n_feature_maps, self.num_classes)
        
        self.gc1 = SAGEConv(self.n_feature_maps, self.n_feature_maps, self.agg_type)
        self.gc2 = SAGEConv(self.n_feature_maps, self.n_feature_maps, self.agg_type)
        self.gc3 = SAGEConv(self.n_feature_maps, self.num_classes, self.agg_type)

    def forward(self, x, graph):
        """
        Propaga os dados através da rede.
        
        Parâmetros
        ----------
        x : torch.Tensor
            Tensor de entrada contendo as séries temporais
        graph : DGLGraph
            O grafo que conecta as amostras
            
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
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        h = F.leaky_relu(self.gc1(graph, x))
        h = F.leaky_relu(self.gc2(graph, h))
        h = self.gc3(graph, h)
        
        return F.log_softmax(h, dim=1)
        
