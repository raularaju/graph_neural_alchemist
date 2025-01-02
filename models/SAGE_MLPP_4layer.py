import torch.nn.functional as F
import torch.nn as nn
import dgl
from dgl.nn.pytorch import SAGEConv
import lightning as pl

class SAGE_MLPP_4layer(pl.LightningModule):
    """
    Implementa um modelo GraphSAGE de 4 camadas com MLP para classificação de grafos.
    
    Esta classe implementa uma rede neural que combina:
    - 4 camadas de convolução GraphSAGE para processamento de grafos
    - Readout por média dos nós para obter representação do grafo 
    - MLP (Perceptron multicamadas) para classificação final
    
    Parâmetros
    ----------
    args : objeto
        Objeto contendo os argumentos de configuração:
        - num_features: Número de características de entrada dos nós
        - nhid: Número de unidades ocultas nas camadas convolucionais
        - num_classes: Número de classes para classificação
        - agg_type: Tipo de agregação usado no GraphSAGE
        - seed: Semente aleatória para reprodutibilidade
    """
    def __init__(self, args):
        super(SAGE_MLPP_4layer, self).__init__()

        self.in_feats = args.num_features
        self.h_feats = args.nhid
        self.num_classes = args.num_classes
        self.agg_type = args.agg_type
        self.seed = args.seed
        
        self.conv1 = SAGEConv(self.in_feats, self.h_feats, self.agg_type)
        self.conv2 = SAGEConv(self.h_feats, self.h_feats, self.agg_type)
        self.conv3 = SAGEConv(self.h_feats, self.h_feats, self.agg_type)
        self.conv4 = SAGEConv(self.h_feats, self.h_feats, self.agg_type)

        # Camadas totalmente conectadas para classificação final
        self.fc1 = nn.Linear(self.h_feats, self.h_feats // 2)
        self.fc2 = nn.Linear(self.h_feats // 2, self.h_feats // 4)
        self.fc3 = nn.Linear(self.h_feats // 4, self.num_classes)

    def forward(self, graph, node_features, edge_weights=None):
        """
        Propaga os dados através da rede, realizando o Message Passing no grafo.
        
        Parâmetros
        ----------
        graph : DGLGraph
            O grafo de entrada
        node_features : torch.Tensor
            Características dos nós do grafo
        edge_weights : torch.Tensor, opcional
            Pesos das arestas do grafo
            
        Retorna
        -------
        torch.Tensor
            Log-probabilidades das classes preditas
        """
        h = F.leaky_relu(self.conv1(graph, node_features, edge_weights))
        h = F.leaky_relu(self.conv2(graph, h, edge_weights))
        h = F.leaky_relu(self.conv3(graph, h, edge_weights))
        h = self.conv4(graph, h, edge_weights)

        with graph.local_scope():
            graph.ndata["h"] = h
            
            readout_h = dgl.mean_nodes(graph, 'h')            
            h = F.leaky_relu(self.fc1(readout_h))
            h = F.leaky_relu(self.fc2(h))
            h = self.fc3(h)

            return F.log_softmax(h, dim=1)