
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
import lightning as pl

class SAGE_NodeClassification(pl.LightningModule):
    """
    Implementa um modelo GraphSAGE para classificação de nós em grafos.
    
    Esta classe implementa uma rede neural que combina:
    - 3 camadas de convolução GraphSAGE para processamento de grafos
    - Uma camada totalmente conectada para classificação final
    """
    def __init__(self, args):
        super(SAGE_NodeClassification, self).__init__()
        
        self.in_feats = args.num_features
        self.n_feature_maps = args.nhid
        self.num_classes = args.num_classes        
        self.agg_type = args.agg_type
        self.seed = args.seed
        
        self.gc1 = SAGEConv(self.in_feats, self.n_feature_maps, self.agg_type)
        self.gc2 = SAGEConv(self.n_feature_maps, self.n_feature_maps, self.agg_type)
        self.gc3 = SAGEConv(self.n_feature_maps, self.num_classes, self.agg_type)

    def forward(self, graph, node_features):
        """
        Realiza o Message Passing no grafo.

        Args:
            graph: O grafo de entrada no formato DGL
            node_features: Tensor com as características dos nós

        Returns:
            Tensor com as probabilidades logarítmicas das classes para cada nó,
            após passar pelas camadas de convolução GraphSAGE e softmax
        """        
        h = F.leaky_relu(self.gc1(graph, node_features))
        h = F.leaky_relu(self.gc2(graph, h))
        h = self.gc3(graph, h)
        
        return F.log_softmax(h, dim=1)