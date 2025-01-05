import os
import torch
import dgl
from dgl.dataloading import GraphDataLoader
from torch.utils.data import DataLoader

from LightningGNN import LightningGNN
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch import Trainer, seed_everything

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from time2graph.run_time2graph_plus import run_time2graphplus

from datasets import *
from models import *
from sim_tsc.models import *

import json
import time
import torch.multiprocessing
import traceback
import logging
from config import get_args
import numpy as np
import warnings

from utils import get_onehotencoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
warnings.filterwarnings("always")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

args = get_args()

ROOT_PATH = args.root_path
TRAIN_PATH = args.train_path
TEST_PATH = args.test_path
# define the logger
if not os.path.exists("logs"):
    os.makedirs("logs")
    
logging.basicConfig(
    level = logging.DEBUG,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%d/%m/%Y %H:%M:%S',
    filename = f"logs/{time.strftime('%Y-%m-%d')}_main.log",
    filemode = 'a'
    
)

main_logger = logging.getLogger(__name__)
main_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
main_logger.addHandler(console_handler)

with open("parameters.json", "r") as f:
    PROJECT_PARAMS = json.load(f)

def main(datasets):
    """
    Função principal que executa experimentos de classificação de séries temporais usando grafos.
    
    Esta função:
    1. Configura reprodutibilidade através de sementes aleatórias para todos os componentes
    2. Configura uso determinístico de GPU e precisão de operações
    3. Para cada dataset:
        - Carrega e pré-processa os dados de treino e teste
        - Constrói grafos usando a estratégia escolhida:
            * "op": Grafos de padrões de transição ordinal
            * "encoded_gtpo": Grafos de padrões de transição ordinal com codificação one-hot
            * "vg": Grafos de visibilidade
            * "simtsc": Matrizes de similaridade para classificação de séries temporais
            * "time2graph": Grafos baseados em shapelets
            * "time2graphplus": Versão estendida do time2graph
        - Configura callbacks para checkpointing e early stopping
        - Treina o modelo GNN especificado usando PyTorch Lightning
        - Avalia no conjunto de teste e registra métricas via TensorBoard
        - Salva predições e relatórios de classificação
    
    Args:
        datasets (list): Lista de nomes dos datasets a serem processados
        
    A função utiliza argumentos globais do argparse incluindo:
        - strategy: Método de construção do grafo
        - model: Arquitetura do modelo GNN
        - batch_size: Tamanho do lote para treinamento
        - epochs: Número de épocas de treinamento
        - early_stopping: Flag para uso de parada antecipada
        - patience: Número de épocas para early stopping
        - seed: Semente para reprodutibilidade
        - save_dir: Diretório para salvar resultados
        - E outros parâmetros configuráveis via config.py
        
    Raises:
        Exception: Registra no log qualquer erro durante a execução de um dataset
    """
    seed_everything(args.seed, workers=True)
    dgl.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    dgl.random.seed(args.seed)
    dgl.seed(args.seed) 
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    torch.multiprocessing.set_sharing_strategy("file_system")
    
    main_logger.info("Using " + str(torch.cuda.device_count()) + " GPU(s)!")
    
    for dataset in datasets:
        inicio = time.time()
        main_logger.info(f"Dataset: {dataset}")
        args.dataset = dataset
        
        try:
            callbacks = [
                ModelCheckpoint(
                    monitor="train_loss", save_top_k=1, mode="min", verbose=True
                ),
                ModelSummary(max_depth=-1),
            ]
            
            if args.early_stopping:
                main_logger.info(f"Using early stopping with patience {args.patience}")
                callbacks.append(EarlyStopping(monitor="train_loss", patience=args.patience))            

            args.train_path = f"{ROOT_PATH}/{dataset}/{dataset}_TRAIN.tsv"
            args.test_path = f"{ROOT_PATH}/{dataset}/{dataset}_TEST.tsv" 
            
            if args.strategy not in PROJECT_PARAMS["valid_strategies"]:
                main_logger.error(f"Time2Graph strategy {args.strategy} não implementada")
                return
            if(args.strategy == "op"):            
                args.dataset_path = f"{ROOT_PATH}/transition_pattern_graphs/{dataset}_oplength_{args.op_length}"                
                dataset_train = GTPODataset(
                    root=os.path.join(
                        args.dataset_path,                    
                        "train"                    
                    ),
                    tsv_file=args.train_path,
                    op_length=args.op_length,
                )

                dataset_test = GTPODataset(
                    root=os.path.join(
                        args.dataset_path,
                        "test"                    
                    ),
                    tsv_file=args.test_path,
                    op_length=args.op_length,
                )
                
                args.num_features = dataset_train.num_features            
                args.num_classes = dataset_train.num_classes
                
                train_dataloader = GraphDataLoader(
                    dataset_train, batch_size=args.batch_size, shuffle=True, ddp_seed=args.seed,
                    num_workers=28
                )
                
                test_dataloader = GraphDataLoader(
                    dataset_test, batch_size=args.batch_size, shuffle=False, ddp_seed=args.seed,
                    num_workers=28
                )            

            elif(args.strategy == "encoded_gtpo"):            
                args.dataset_path = f"{ROOT_PATH}/transition_pattern_graphs_encoded/{dataset}_oplength_{args.op_length}"
                
                if not os.path.exists(os.path.join(args.dataset_path, "train")):                
                    encoder = get_onehotencoder(args.train_path, args.test_path, args.op_length)
                else:
                    encoder = OneHotEncoder(sparse_output=False)
                
                dataset_train = encodedGTPODataset(
                    root=os.path.join(
                        args.dataset_path,                    
                        "train"                    
                    ),
                    encoder=encoder,
                    tsv_file=args.train_path,
                    op_length=args.op_length,
                )

                dataset_test = encodedGTPODataset(
                    root=os.path.join(
                        args.dataset_path,
                        "test"                    
                    ),
                    tsv_file=args.test_path,
                    encoder=encoder,
                    op_length=args.op_length,
                )
                
                args.num_features = dataset_train.num_features            
                args.num_classes = dataset_train.num_classes
                
                train_dataloader = GraphDataLoader(
                    dataset_train, batch_size=args.batch_size, shuffle=True, ddp_seed=args.seed,
                    num_workers=28
                )
                
                test_dataloader = GraphDataLoader(
                    dataset_test, batch_size=args.batch_size, shuffle=False, ddp_seed=args.seed,
                    num_workers=28
                )            
            
            elif(args.strategy == "vg"):
                args.dataset_path = f"{ROOT_PATH}/visibility_graphs/signal_as_feat/{dataset}"
                dataset_train = VisibiliyGraphDataset(
                    root=os.path.join(
                        args.dataset_path,                    
                        "train"                    
                    ),
                    tsv_file=args.train_path,
                    directed="left_to_right",
                    weighted="sq_distance",
                )
                
                dataset_test = VisibiliyGraphDataset(
                    root=os.path.join(
                        args.dataset_path,
                        "test"                    
                    ),
                    tsv_file=args.test_path,
                    directed="left_to_right",
                    weighted="sq_distance",
                )
                
                args.num_features = dataset_train.num_features            
                args.num_classes = dataset_train.num_classes
                
                train_dataloader = GraphDataLoader(
                    dataset_train, batch_size=args.batch_size, shuffle=True, ddp_seed=args.seed,
                    num_workers=28
                )
                
                test_dataloader = GraphDataLoader(
                    dataset_test, batch_size=args.batch_size, shuffle=False, ddp_seed=args.seed,
                    num_workers=28
                )
            
            elif(args.strategy in ["simtsc"]):
                if(args.model not in ["simTSC_GCN", "simTSC_SAGE"]):
                    main_logger.error(f"Modelo {args.model} não implementado para estratégia {args.strategy}. Favor usar simTSC_GCN ou simTSC_SAGE")
                    return
                
                args.dataset_path = f"{ROOT_PATH}/simtsc/{args.strategy}_matrix/{dataset}"               

                dataset_train = TimeSeriesDataset(                    
                    tsv_file=args.train_path,
                )
                dataset_test = TimeSeriesDataset(
                    tsv_file=args.test_path,
                )
                                            
                args.num_features = dataset_train.num_features            
                args.num_classes = dataset_train.num_classes
                
                train_dataloader = DataLoader(
                    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=28
                )
                test_dataloader = DataLoader(
                    dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=28
                )    
                
            elif(args.strategy == "pearson"):
                args.dataset_path = f"{ROOT_PATH}/pearson/{dataset}"
                args.R = 8
                dataset_train = CovarianceGraphDataset(
                    root=os.path.join(
                        args.dataset_path,                    
                        "train"                    
                    ),
                    tsv_file=args.train_path,
                    R=args.R,
                )
                
                dataset_test = CovarianceGraphDataset(
                    root=os.path.join(
                        args.dataset_path,
                        "test"                    
                    ),
                    tsv_file=args.test_path,
                    R=args.R,
                )
                
                args.num_features = dataset_train.num_features            
                args.num_classes = dataset_train.num_classes
                
                train_dataloader = GraphDataLoader(
                    dataset_train, batch_size=args.batch_size, shuffle=True, ddp_seed=args.seed,
                    num_workers=28
                )
                
                test_dataloader = GraphDataLoader(
                    dataset_test, batch_size=args.batch_size, shuffle=False, ddp_seed=args.seed,
                    num_workers=28
                )

            elif(args.strategy == "time2graph"):
                args.dataset_path = f"{ROOT_PATH}/time2graph/{dataset}"
                args.dataset = dataset
                args.K = 50
                args.C = 800
                args.seg_length = 24
                args.num_segment = 10
                # args.time_aware_shapelets_lr = 0.1
                args.percentile = 80
                args.alpha = 0.1
                args.beta = 0.05                
                
                dataset_train = Time2GraphDataset(
                    root=os.path.join(
                        args.dataset_path,                    
                        "train"                    
                    ),
                    tsv_file=args.train_path,
                    args=args,
                )
                
                dataset_test = Time2GraphDataset(
                    root=os.path.join(
                        args.dataset_path,
                        "test"                    
                    ),
                    tsv_file=args.test_path,
                    args=args,
                )
                
                args.num_features = dataset_train.num_features            
                args.num_classes = dataset_train.num_classes
                
                train_dataloader = GraphDataLoader(
                    dataset_train, batch_size=args.batch_size, shuffle=True, ddp_seed=args.seed,
                    num_workers=28
                )
                
                test_dataloader = GraphDataLoader(
                    dataset_test, batch_size=args.batch_size, shuffle=False, ddp_seed=args.seed,
                    num_workers=28
                )
            
            elif(args.strategy == "time2graphplus"):
                args.dataset_path = f"{ROOT_PATH}/time2graphplus/{dataset}"
                args.K = 50
                args.C = 800
                args.seg_length = 24
                args.num_segment = 10
                args.time_aware_shapelets_lr = 0.1
                args.percentile = 80
                args.alpha = 0.1
                args.beta = 0.05
                args.save_dir = os.path.join("lightning_logs", args.save_dir, dataset)               
                
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                
                init_time = time.time()
                run_time2graphplus(args)               
                
                print("Tempo de execução: ", time.time() - init_time)
                continue
            
            if(args.model == None):
                main_logger.error(f"Modelo não definido")
                return
            
            model = eval(args.model)(args)

            logs_name_path = os.path.join(dataset, args.model)
            
            tb_logger = TensorBoardLogger(
                save_dir=os.path.join("lightning_logs", args.save_dir),
                name=logs_name_path,
                log_graph=False,
                default_hp_metric=False,                               
            )
            
            tb_logger.log_hyperparams(vars(args))
            
            fast_dev_run = args.fast_dev_run_batches if args.fast_dev_run else False
            
            trainer = Trainer(
                callbacks=callbacks,
                logger=tb_logger,
                max_epochs=args.epochs,
                detect_anomaly=args.detect_anomaly,
                fast_dev_run=fast_dev_run,
                overfit_batches=args.overfit_batches,
                deterministic=True,
                log_every_n_steps = 20,
            )

            modulo = LightningGNN(args, model)            
            trainer.fit(modulo, train_dataloaders=train_dataloader)
            
            if not args.fast_dev_run:                           
                trainer.test(dataloaders=test_dataloader, verbose=True, ckpt_path= "best")
            
            main_logger.info(f"Tempo de execução: {(time.time() - inicio):.2f} segundos")
            
        except Exception as e:            
            main_logger.error(f"Erro ao executar o dataset {dataset}: {e.__str__}:")
            main_logger.error(traceback.format_exc())


if __name__ == "__main__":     
    datasets = PROJECT_PARAMS["datasets"]
    main(datasets)
