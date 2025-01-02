from .graphConstruction import construct_graph_dtw, construct_graph_pearson


def run(batch_idx, labels, data, args, type_step):
    """
    Executa a construção do grafo de acordo com a estratégia escolhida.
    
    Esta função:
    1. Verifica o tipo de modelo para determinar se retorna apenas a matriz de adjacência
    2. Constrói o grafo usando correlação de Pearson ou DTW de acordo com args.strategy
    
    Parâmetros
    ----------
    batch_idx : int
        Índice do lote atual
    labels : torch.Tensor
        Tensor contendo os rótulos das séries
    data : torch.Tensor
        Tensor contendo as séries temporais
    args : objeto
        Objeto contendo os argumentos de configuração:
        - model: Tipo de modelo (simTSC_GCN ou outro)
        - strategy: Estratégia de construção do grafo (pearson ou simtsc)
        - R: Limiar de correlação para estratégia pearson
        - K: Número de vizinhos para estratégia DTW
        - alpha: Parâmetro de escala para DTW
        - dataset_path: Caminho para salvar/carregar matrizes
    type_step : str
        Identificador do passo (treino/validação/teste)
        
    Retorna
    -------
    Union[dgl.DGLGraph, torch.sparse.FloatTensor]
        Grafo DGL ou matriz de adjacência esparsa, dependendo do modelo
    """
    adj_only = True if args.model =="simTSC_GCN" else False            
    if args.strategy == "pearson":
        graph = construct_graph_pearson(time_series=data,
                                        labels=labels,
                                        R=args.R,
                                        type_step=type_step,
                                        batch_idx=batch_idx,
                                        dataset_path=args.dataset_path,
                                        adj_only=adj_only)
    elif args.strategy == "simtsc":
        graph = construct_graph_dtw(time_series=data,
                                    labels=labels,
                                    K=args.K,
                                    alpha=args.alpha,
                                    type_step=type_step,
                                    batch_idx=batch_idx,
                                    dataset_path=args.dataset_path,
                                    adj_only=adj_only)
    return graph   