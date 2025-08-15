from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from dgl.dataloading import GraphDataLoader
import torch
import numpy as np
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.classification import MulticlassF1Score
import copy
import os
from datasets import *
from models import *
from utils import get_onehotencoder
from sklearn.preprocessing import OneHotEncoder
import optuna
from config import get_args
from LightningGNN import LightningGNN
from sklearn.metrics import f1_score 
from sklearn.model_selection import StratifiedShuffleSplit
import dgl

def get_model_hyperparams(trial, model_name):
    """Define espaço de busca para hiperparâmetros do modelo."""
    
    if model_name == "SAGE_MLPP_4layer":
        return {
            "nhid": trial.suggest_categorical("nhid", [32, 64, 128, 256]),
            "agg_type": trial.suggest_categorical("agg_type", ["mean", "pool", "lstm", "gcn"]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        }
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")


def get_dataset_hyperparams(trial, strategy):
    """Define espaço de busca para hiperparâmetros do dataset/estratégia."""
    
    if strategy == "op":
        return {
            "op_length": trial.suggest_int("op_length", 3, 7)
        }
    
    elif strategy == "encoded_gtpo":
        return {
            "op_length": trial.suggest_int("op_length", 3, 7)
        }
    
    elif strategy == "vg":
        return {}
    
    elif strategy == "pearson":
        return {
            "R": trial.suggest_int("R", 4, 16, step=2)
        }
    
    elif strategy == "time2graph":
        return {
            "K": trial.suggest_int("K", 30, 100, step=10),
            "C": trial.suggest_int("C", 500, 1000, step=100),
            "seg_length": trial.suggest_int("seg_length", 16, 32, step=4),
            "num_segment": trial.suggest_int("num_segment", 5, 15),
            "percentile": trial.suggest_int("percentile", 70, 90, step=5),
            "alpha": trial.suggest_float("alpha", 0.05, 0.2, step=0.05),
            "beta": trial.suggest_float("beta", 0.01, 0.1, step=0.01)
        }
    
    elif strategy in ["simtsc"]:
        return {}  # simtsc não tem hiperparâmetros específicos de dataset
    
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")


def create_datasets_with_hyperparams(base_args, dataset_hyperparams, trial_id):
    """Cria os datasets de treino e teste com os hiperparâmetros sugeridos."""
    
    # Cria uma cópia dos args para não modificar o original
    args = copy.deepcopy(base_args)
    
    # Aplica hiperparâmetros do dataset
    for param, value in dataset_hyperparams.items():
        setattr(args, param, value)
    
    # Atualiza paths se necessário
    if args.strategy == "op":
        args.dataset_path = f"{args.root_path}/transition_pattern_graphs/{args.dataset}_oplength_{args.op_length}"
    elif args.strategy == "encoded_gtpo":
        args.dataset_path = f"{args.root_path}/transition_pattern_graphs_encoded/{args.dataset}_oplength_{args.op_length}"
    elif args.strategy == "pearson":
        args.dataset_path = f"{args.root_path}/pearson/{args.dataset}"
    elif args.strategy == "time2graph":
        args.dataset_path = f"{args.root_path}/time2graph/{args.dataset}"
    
    # Cria datasets baseado na estratégia
    if args.strategy == "op":
        dataset_train = GTPODataset(
            root=os.path.join(args.dataset_path, "train"),
            tsv_file=args.train_path,
            op_length=args.op_length,
        )
        dataset_test = GTPODataset(
            root=os.path.join(args.dataset_path, "test"),
            tsv_file=args.test_path,
            op_length=args.op_length,
        )
        
    elif args.strategy == "encoded_gtpo":
        if not os.path.exists(os.path.join(args.dataset_path, "train")):
            encoder = get_onehotencoder(args.train_path, args.test_path, args.op_length)
        else:
            encoder = OneHotEncoder(sparse_output=False)
            
        dataset_train = encodedGTPODataset(
            root=os.path.join(args.dataset_path, "train"),
            tsv_file=args.train_path,
            encoder=encoder,
            op_length=args.op_length,
        )
        dataset_test = encodedGTPODataset(
            root=os.path.join(args.dataset_path, "test"),
            tsv_file=args.test_path,
            encoder=encoder,
            op_length=args.op_length,
        )
    elif args.strategy == "vg":
        print(f"\n\n Train path: {args.train_path} \n\n")
        dataset = PreComputedVGDataset(
            root=args.dataset_path,
            tsv_file=args.train_path,
            node_features_file=args.node_features_train_path,
            graphs_folder=args.graphs_train_folder,
            dataset_name=args.dataset,
        )
        
        
    elif args.strategy == "pearson":
        dataset_train = CovarianceGraphDataset(
            root=os.path.join(args.dataset_path, "train"),
            tsv_file=args.train_path,
            R=args.R,
        )
        dataset_test = CovarianceGraphDataset(
            root=os.path.join(args.dataset_path, "test"),
            tsv_file=args.test_path,
            R=args.R,
        )
        
    elif args.strategy == "time2graph":
        dataset_train = Time2GraphDataset(
            root=os.path.join(args.dataset_path, "train"),
            tsv_file=args.train_path,
            args=args,
        )
        dataset_test = Time2GraphDataset(
            root=os.path.join(args.dataset_path, "test"),
            tsv_file=args.test_path,
            args=args,
        )
    elif args.strategy in ["simtsc"]:
        dataset_train = TimeSeriesDataset(tsv_file=args.train_path)
        dataset_test = TimeSeriesDataset(tsv_file=args.test_path)
    else:
        raise ValueError(f"Estratégia não suportada: {args.strategy}")
    
    # Atualiza args com informações do dataset
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    
    return dataset, args


def create_dataloaders(dataset, batch_size, shuffle=True, strategy="vg"):
    """Cria dataloaders apropriados para cada estratégia."""
    
    if strategy in ["simtsc"]:
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=4
        )
    else:
        with open(args.train_path, "r") as file:
            lines = file.readlines()

        y_all = np.array([int(line.split("\t")[0]) for line in lines])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
        for _, sample_indices in sss.split(np.zeros(len(y_all)), y_all):
            subset = Subset(dataset, sample_indices)
        return GraphDataLoader(
            subset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            ddp_seed=args.seed,
            num_workers=4
        )


def evaluate_model_cv(model_class, dataset, args, cv_folds=5):
    """Avalia modelo usando cross-validation no dataset de treino."""
    # Extrai labels do dataset
    y_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        y_labels.append(label.item() if torch.is_tensor(label) else label)
    
    y_labels = np.array(y_labels)
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), y_labels)):
        print(f"  Fold {fold + 1}/{cv_folds}")
        
        # Cria subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # Cria dataloaders
        train_loader = create_dataloaders(train_subset, args.batch_size, True, args.strategy)
        val_loader = create_dataloaders(val_subset, args.batch_size, False, args.strategy)
        
        # Instancia modelo
        model = eval(model_class)(args)
        
        logs_name_path = os.path.join(args.dataset, args.model)
        
        tb_logger = TensorBoardLogger(
                save_dir=os.path.join("lightning_logs", args.save_dir),
                name=logs_name_path,
                log_graph=False,
                default_hp_metric=False,                               
            )
        # Trainer para CV (menos épocas para speed)
        trainer = Trainer(
            max_epochs=min(args.epochs, 50),  # Limita épocas para CV
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=tb_logger,
            deterministic=True,
        )
        
        # Treina
        modulo = LightningGNN(args, model)
        trainer.fit(modulo, train_dataloaders=train_loader)
        
        # Avalia
        modulo.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                #if args.strategy in ["simtsc"]:
                graphs, labels = batch
                node_features = graphs.ndata['feat']
                preds = modulo.model(graphs, node_features)
               
                preds = preds.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        # Calcula F1-macro
        f1 = f1_score(
                all_labels.numpy(),  # Converte para numpy 
                all_preds.numpy(),   # Converte para numpy
                average='macro'      # F1-macro como você usou no RandomizedSearchCV
            )
        cv_scores.append(f1)
        print(f"    F1-macro: {f1:.4f}")
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    print(f"  CV F1-macro: {mean_score:.4f} ± {std_score:.4f}")
    
    return mean_score


def create_objective_complete(base_args, model_class, strategy):
    """Cria função objetivo que otimiza hiperparâmetros do modelo E dataset."""
    
    def objective(trial):
        print(f"\nTrial {trial.number}")
        
        try:
            # Obtém hiperparâmetros do modelo
            model_hyperparams = get_model_hyperparams(trial, model_class)
            
            # Obtém hiperparâmetros do dataset
            dataset_hyperparams = get_dataset_hyperparams(trial, strategy)
            
            # Cria args combinados
            trial_args = copy.deepcopy(base_args)
            for param, value in model_hyperparams.items():
                setattr(trial_args, param, value)
            for param, value in dataset_hyperparams.items():
                setattr(trial_args, param, value)
            
            # Cria datasets com novos hiperparâmetros
            dataset, trial_args = create_datasets_with_hyperparams(
                trial_args, dataset_hyperparams, trial.number
            )
            
            # Avalia com cross-validation
            cv_score = evaluate_model_cv(
                model_class, 
                dataset, 
                trial_args,
                cv_folds=5
            )
            
            print(f"Score final: {cv_score:.4f}")
            return cv_score
            
        except Exception as e:
            print(f"Erro no trial {trial.number}: {e}")
            import traceback
            traceback.print_exc()
            return 0.0  # Retorna score baixo em caso de erro
    
    return objective


# Função principal para executar busca
def run_hyperparameter_search(base_args, model_class, strategy, n_trials=30):
    """Executa busca de hiperparâmetros completa."""
    
    print(f"Iniciando busca de hiperparâmetros")
    print(f"Estratégia: {strategy}")
    print(f"Modelo: {model_class}")
    print(f"Número de trials: {n_trials}")
    
    # Cria estudo Optuna
    study = optuna.create_study(
        direction='maximize',
        study_name=f"{base_args.dataset}_{strategy}_{model_class}",
        storage=None,  # Pode configurar para salvar em DB
        load_if_exists=True
    )
    
    # Cria função objetivo
    objective_func = create_objective_complete(base_args, model_class, strategy)
    
    # Executa otimização
    study.optimize(objective_func, n_trials=n_trials, timeout=None)
    
    # Resultados
    print(f"\nMelhores hiperparâmetros encontrados:")
    print(f"F1-macro: {study.best_value:.4f}")
    print(f"Parâmetros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params, study.best_value


if __name__ == "__main__":
    # Exemplo de uso
    
    args = get_args()
    
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
    # Configura parâmetros base
    args.dataset = "ECG200"
    args.strategy = "vg"  
    args.epochs = 250
    ROOT_PATH = args.root_path
    args.dataset_path = f"{ROOT_PATH}/visibility_graphs/signal_as_feat/{args.dataset}"
    args.train_path = f"{ROOT_PATH}/{args.dataset}/{args.dataset}_MERGED.tsv"
    
    # Executa busca
    best_params, best_score = run_hyperparameter_search(
        args, 
        "SAGE_MLPP_4layer", 
        "vg", 
        n_trials=1
    )
    
    print(f"\nBusca finalizada!")
    print(f"Melhor F1-macro: {best_score:.4f}")
    print(f"Melhores parâmetros: {best_params}")