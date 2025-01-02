import argparse
import warnings
from pathos.helpers import mp
from .core.time2graph_model import Time2Graph
from .load_usr_dataset import load_usr_dataset_by_name
from .utils.base_utils import Debugger
import numpy as np
from os import path, makedirs
from sklearn.metrics import matthews_corrcoef, classification_report

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# -*- coding: utf-8 -*-

warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=RuntimeWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)


def run_time2graphplus(args):
    """
    Executa o modelo Time2GraphPlus para classificação de séries temporais.
    
    Args:
        args: Objeto contendo os argumentos de configuração do modelo, incluindo:
            - data_size: Tamanho dos dados (fixado em 1)
            - percentile: Percentil para seleção de shapelets (fixado em 80)
            - diff: Flag para usar diferenciação (False)
            - standard_scale: Flag para normalização (False) 
            - softmax: Flag para usar softmax (False)
            - append: Flag para anexar features (False)
            - sort: Flag para ordenar (False)
            - aggregate: Flag para agregação (False)
            - feat_flag: Flag para features (False)
            - feat_norm: Flag para normalização de features (False)
            - debug: Flag para modo debug (False)
            - init: Valor inicial (0)
            - warp: Fator de warping (2)
            - tflag: Flag para time-aware (True)
            - mode: Modo de operação ("embedding")
            - candidate_method: Método de seleção de candidatos ("greedy")
            - measurement: Medida de distância ("gdtw")
            - gpu_enable: Flag para uso de GPU (True)
            - dataset: Nome do dataset extraído do caminho de treino
            - train_path: Caminho para dados de treino
            - test_path: Caminho para dados de teste
            - num_segment: Número de segmentos
            - seg_length: Tamanho dos segmentos
            - K: Número de shapelets
            - dataset_path: Caminho base do dataset
            - save_dir: Diretório para salvar resultados
    """
    #verify if start_method has been set, if not, set it to spawn
    if not mp.get_start_method(allow_none=True):        
        mp.set_start_method('spawn')
    
    args.data_size = 1
    args.percentile = 80
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
    args.gpu_enable= True    
    args.dataset = args.train_path.split("/")[-1].split(".")[0]    
    root_path = args.dataset_path
    results_path = args.save_dir
    
    Debugger.info_print(f"Running Time2GraphPlus...")
    
    #if results path does not exist, create it
    if not path.isdir(results_path):
        makedirs(results_path, exist_ok=True)        

    if path.isfile(f"{root_path}/cache/{args.dataset}_x_train.npy"):
        x_train = np.load(f"{root_path}/cache/{args.dataset}_x_train.npy")
        labels_train = np.load(f"{root_path}/cache/{args.dataset}_y_train.npy")
        
        x_test = np.load(f"{root_path}/cache/{args.dataset}_x_test.npy")
        labels_test = np.load(f"{root_path}/cache/{args.dataset}_y_test.npy")
        
    else:
        x_train, labels_train= load_usr_dataset_by_name(
            tsv_file= args.train_path,
            num_segment= args.num_segment,
            seg_length= args.seg_length,
        )
        
        x_test, labels_test= load_usr_dataset_by_name(
            tsv_file= args.test_path,
            num_segment= args.num_segment,
            seg_length= args.seg_length,
        )
    
    args.seg_length = x_train.shape[1] // args.num_segment
    args.num_classes = len(np.unique(labels_train))

    Debugger.info_print(f"original training shape {x_train.shape}")
    Debugger.info_print(
        f"basic statistics: max {np.max(x_train):.4f}, min {np.min(x_train):.4f}"
    )
    Debugger.info_print(
        f"in this split: training {len(x_train)} samples, with {sum(labels_train)} positive"
    )
    
    shapelet_cache = f"{root_path}/cache/{args.dataset}_{args.K}_{args.seg_length}_time_aware_shapelets.cache"
        
    if not path.isfile(shapelet_cache):
        makedirs(f"{root_path}/cache", exist_ok=True)
    
    model = Time2Graph(args)
    
    if path.isfile(shapelet_cache):
        model.load_shapelets(fpath=shapelet_cache)
        Debugger.info_print(f"load shapelets from {shapelet_cache}")
    else:
        Debugger.info_print(f"train_size {x_train.shape}, label size {labels_train.shape}")
        model.learn_shapelets(
            x=x_train,
            y=labels_train,
            num_segment=args.num_segment,
            data_size=args.data_size,
        )
        Debugger.info_print(f"learn time-aware shapelets done...")
        model.save_shapelets(shapelet_cache)

    Debugger.info_print(
        f"training: {float(sum(labels_train) / len(labels_train)):.2f} positive ratio with {len(labels_train)}"
    )
    
    model.fit(X=x_train, Y=labels_train)
    
    y_pred = model.predict(X=x_test)
    
    #store the y_preds on the results path
    np.save(f"{results_path}/{args.dataset}_y_pred.npy", y_pred)
    
    report = classification_report(y_true=labels_test, y_pred=y_pred, output_dict=True)
    mcc = matthews_corrcoef(y_true=labels_test, y_pred=y_pred)
    
    Debugger.info_print(f"Classification report: {report}")
    Debugger.info_print(f"Matthews correlation coefficient: {mcc}")

    model_cache = f"{root_path}/cache/{args.dataset}_ttgp_time-aware.cache"    
    model.save_model(model_cache)
    
    #save the values for classification report and mcc into a formatted .csv file
    with open(f"{results_path}/{args.dataset}_classification_report.csv", "w") as f:
        print(f"saving classification report to {results_path}/{args.dataset}_classification_report.csv")
        f.write("metric, value\n")
        for key, value in report.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    f.write(f"{key}_{k}, {v}\n")
            else:
                f.write(f"{key}, {value}\n")
        f.write(f"mcc, {mcc}\n")   
