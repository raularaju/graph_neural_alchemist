import torch.nn.functional as F
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GATv2Conv, GATConv
import lightning as pl


class GAT_MLPP(pl.LightningModule):
    """
    Implementa um modelo GAT com MLP para classificação de grafos.
    
    Esta classe implementa uma rede neural que combina:
    - 3 camadas de convolução GAT para processamento de grafos
    - Uma camada totalmente conectada para classificação final
    """
    def __init__(self, args):
        super(GAT_MLPP, self).__init__()

        self.in_feats = args.num_features
        self.h_feats = args.nhid
        self.num_classes = args.num_classes
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers

        # Define as camadas GAT
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(self.in_feats, self.h_feats, self.num_heads))
        
        for _ in range(self.num_layers - 1):
            self.gat_layers.append(GATConv(self.h_feats * self.num_heads, self.h_feats, self.num_heads))

        # Define as camadas totalmente conectadas
        self.fc1 = nn.Linear(self.h_feats * self.num_heads, self.h_feats // 2)
        self.fc2 = nn.Linear(self.h_feats // 2, self.h_feats // 4)
        self.fc3 = nn.Linear(self.h_feats // 4, self.num_classes)
        

    def forward(self, graph, node_features):
        """
        Propaga os dados através da rede, realizando o Message Passing no grafo.
        
        Parâmetros
        ----------
        graph : DGLGraph
            O grafo de entrada
        node_features : torch.Tensor
            Características dos nós do grafo
        
        Retorna
        -------
        tuple
            Uma tupla contendo:
            - Tensor com as probabilidades logarítmicas das classes para cada nó
            - O grafo com os dados propagados
        """
        h = node_features
        for gat_layer in self.gat_layers:
            h = F.leaky_relu(gat_layer(graph, h).flatten(1))

        with graph.local_scope():
            graph.ndata['h'] = h
            readout_h = dgl.mean_nodes(graph, 'h') 

            h = F.leaky_relu(self.fc1(readout_h))
            h = F.leaky_relu(self.fc2(h))
            h = self.fc3(h)
            return F.log_softmax(h, dim=1), graph
