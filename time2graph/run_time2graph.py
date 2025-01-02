import argparse
import warnings
from .core.time2graph_model import Time2Graph
from .load_usr_dataset import load_usr_dataset_by_name
from .utils.base_utils import Debugger
import numpy as np
from pathos.helpers import mp
from os import path, makedirs
import torch
# -*- coding: utf-8 -*-

warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=RuntimeWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)


def run_time2graph(args):
    """
    Executa o algoritmo Time2Graph para converter séries temporais em representações de grafo.

    Esta função:
    1. Configura os parâmetros do modelo
    2. Carrega ou gera os dados de treinamento
    3. Treina o modelo para extrair shapelets sensíveis ao tempo
    4. Gera representações em grafo das séries temporais

    Parâmetros:
        args: Objeto contendo os argumentos de configuração do modelo, incluindo:
            - train_path: Caminho do arquivo de treinamento
            - root_path: Diretório raiz para cache
            - num_segment: Número de segmentos para dividir a série
            - seg_length: Comprimento de cada segmento
            - K: Número de shapelets a serem extraídos
            - outros parâmetros do modelo

    Retorna:
        tupla: Contendo:
            - node_features: Features dos nós do grafo (numpy.ndarray)
            - adj_matrix: Matriz de adjacência do grafo (numpy.ndarray)
            - labels: Rótulos das séries temporais (numpy.ndarray)
    """
    if not mp.get_start_method(allow_none=True):        
        mp.set_start_method('spawn')

    args.data_size = 1
    args.diff = False
    args.standard_scale = False
    args.softmax = False
    args.append = False
    args.sort = False
    args.aggregate = False
    args.feat_flag = False
    args.feat_norm = False
    args.debug = False    
    args.init = 0
    args.warp = 2
    args.tflag = True
    args.mode = "embedding"
    args.candidate_method = "greedy"
    args.measurement = "gdtw"    
    args.gpu_enable= True if torch.cuda.is_available() else False    
    args.dataset = args.train_path.split("/")[-1].split(".")[0]
    root_path = args.root_path

    if path.isfile(f"{root_path}/cache/{args.dataset}_x_train.npy"):
        x_train = np.load(f"{root_path}/cache/{args.dataset}_x_train.npy")
        labels = np.load(f"{root_path}/cache/{args.dataset}_y_train.npy")
    else:
        x_train, labels= load_usr_dataset_by_name(
            tsv_file= args.tsv_file,
            num_segment= args.num_segment,
            seg_length= args.seg_length,
        )
    
    args.seg_length = x_train.shape[1] // args.num_segment
    args.num_classes = len(np.unique(labels))

    Debugger.info_print(f"original training shape {x_train.shape}")
    Debugger.info_print(
        f"basic statistics: max {np.max(x_train):.4f}, min {np.min(x_train):.4f}"
    )
    Debugger.info_print(
        f"in this split: training {len(x_train)} samples, with {sum(labels)} positive"
    )
    
    shapelet_cache = f"{root_path}/cache/time2graphplus_{args.K}_{args.seg_length}_time_aware_shapelets.cache"
        
    if not path.isfile(shapelet_cache):
        makedirs(f"{root_path}/cache", exist_ok=True)
    
    model = Time2Graph(args)
    
    if path.isfile(shapelet_cache):
        model.load_shapelets(fpath=shapelet_cache)
        Debugger.info_print(f"load shapelets from {shapelet_cache}")
    else:
        Debugger.info_print(f"train_size {x_train.shape}, label size {labels.shape}")
        model.learn_shapelets(
            x=x_train,
            y=labels,
            num_segment=args.num_segment,
            data_size=args.data_size,
        )
        Debugger.info_print(f"learn time-aware shapelets done...")
        model.save_shapelets(shapelet_cache)

    Debugger.info_print(
        f"training: {float(sum(labels) / len(labels)):.2f} positive ratio with {len(labels)}"
    )

    X_scale = model.preprocess_input_data(x_train)
    node_features, adj_matrix = model.__gat_features__(X_scale, train=True)
    return node_features, adj_matrix, labels