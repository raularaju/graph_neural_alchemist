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
import json


NUM_TRIALS = 30
MAX_EPOCHS = 250
N_FOLDS = 5

def get_model_hyperparams(trial, model_name):
    """Define espaço de busca para hiperparâmetros do modelo."""
    
    if model_name == "SAGE_MLPP_4layer":
        return {
            "nhid": trial.suggest_categorical("nhid", [32, 64, 128, 256]),
            "agg_type": trial.suggest_categorical("agg_type", ["mean", "pool", "lstm", "gcn"]),
            "lr": trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        }
    elif model_name == "GAT_MLPP":
        return {
            "nhid": trial.suggest_categorical("nhid", [32, 64, 128, 256]),
            "num_heads": trial.suggest_categorical("num_heads", [1, 2, 3]),
            "num_layers": trial.suggest_categorical("num_layers", [1, 2, 3]),
        }
    elif model_name == "simTSC_GCN":
        return {
            "nhid": trial.suggest_categorical("nhid", [16, 32, 64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=1),
        }
    elif model_name == "simTSC_SAGE":
        return {
            "nhid": trial.suggest_categorical("nhid", [32, 64, 128, 256]),
            "agg_type": trial.suggest_categorical("agg_type", ["mean", "pool", "lstm", "gcn"]),
        }
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")


def get_dataset_hyperparams(trial, strategy):
    """Define espaço de busca para hiperparâmetros do dataset/estratégia."""
    
    if strategy == "op":
        return {
            "op_length": trial.suggest_int("op_length", 4, 8)
        }
    
    elif strategy == "encoded_gtpo":
        return {
            "op_length": trial.suggest_int("op_length", 4, 8)
        }
    
    elif strategy == "vg":
        return {}
    
    elif strategy == "simtsc":
        return {} 

    elif strategy == "pearson":
        return {
            "R": trial.suggest_float("R", 0.5, 0.9, step=0.05)
        }

    elif strategy == "time2graph":
        return {
            "K": trial.suggest_int("K", 30, 100, step=10),
            "C": trial.suggest_int("C", 500, 1000, step=100),
            "seg_length": trial.suggest_int("seg_length", 16, 32, step=4),
            "num_segment": trial.suggest_int("num_segment", 5, 15),
            "percentile": trial.suggest_int("percentile", 70, 90, step=5),
            "alpha": trial.suggest_float("alpha", 0.1, 0.4, step=0.05),
            "beta": trial.suggest_float("beta", 0.01, 0.1, step=0.01)
        }
    
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
        dataset = GTPODataset(
            root=args.dataset_path,
            tsv_file=args.train_path,
            op_length=args.op_length,
        )
        
    elif args.strategy == "encoded_gtpo":
        encoder = OneHotEncoder(sparse_output=False)
            
        dataset = encodedGTPODataset(
            root=args.dataset_path,
            tsv_file=args.train_path,
            encoder=encoder,
            op_length=args.op_length,
        )

    elif args.strategy == "vg":
        dataset = PreComputedVGDataset(
            root=args.dataset_path,
            tsv_file=args.train_path,
            node_features_file=args.node_features_train_path,
            graphs_folder=args.graphs_train_folder,
            dataset_name=args.dataset,
        )
        
        
    elif args.strategy == "pearson":
        dataset = CovarianceGraphDataset(
            root=args.dataset_path,
            tsv_file=args.train_path,
            R=args.R,
        )
        
    elif args.strategy == "time2graph":
        dataset = Time2GraphDataset(
            root=args.dataset_path,
            tsv_file=args.train_path,
            args=args,
        )

    elif args.strategy == "simtsc":
        dataset = TimeSeriesDataset(tsv_file=args.train_path)

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
        return GraphDataLoader(
            dataset, 
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
  
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
    for _, sample_indices in sss.split(np.zeros(len(y_labels)), y_labels):
        cv_subset = Subset(dataset, sample_indices)
        y_labels = y_labels[sample_indices]
    # Cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(cv_subset)), y_labels)):
        print(f"  Fold {fold + 1}/{cv_folds}")
        
        # Cria subsets
        train_subset = Subset(cv_subset, train_idx)
        val_subset = Subset(cv_subset, val_idx)
        
        # Cria dataloaders
        train_loader = create_dataloaders(train_subset, args.batch_size, True, args.strategy)
        val_loader = create_dataloaders(val_subset, args.batch_size, False, args.strategy)
        
        # Instancia modelo
        model = eval(model_class)(args)
        
        logs_name_path = os.path.join(args.dataset, args.model)
        
        tb_logger = TensorBoardLogger(
                save_dir=os.path.join("lightning_logs","hyperparam", args.save_dir),
                name=logs_name_path,
                log_graph=False,
                default_hp_metric=False,                               
            )
        # Trainer para CV (menos épocas para speed)
        trainer = Trainer(
            max_epochs=min(args.epochs, MAX_EPOCHS),  # Limita épocas para CV
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
                cv_folds=N_FOLDS
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
def run_hyperparameter_search(base_args, n_trials=NUM_TRIALS):
    """Executa busca de hiperparâmetros completa."""
    

    model_class = base_args.model
    strategy = base_args.strategy
    print(f"Iniciando busca de hiperparâmetros")
    print(f"Estratégia: {strategy}")
    print(f"Modelo: {model_class}")
    print(f"Número de trials: {n_trials}")
    
    # Cria estudo Optuna
    study = optuna.create_study(
        direction='maximize',
        study_name=f"{base_args.dataset}_{strategy}_{model_class}",
        storage=None,  # Pode configurar para salvar em DB
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=base_args.seed)
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


def setup(args):
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

if __name__ == "__main__":
    # Exemplo de uso
    
    args = get_args()
    
    setup(args)

    # Configura parâmetros base
    ROOT_PATH = args.root_path
    args.dataset_path = f"{ROOT_PATH}/visibility_graphs/signal_as_feat/{args.dataset}"
    args.train_path = f"{ROOT_PATH}/{args.dataset}/{args.dataset}_MERGED.tsv"
    
    # Executa busca
    best_params, best_score = run_hyperparameter_search(
        args, 
        n_trials=NUM_TRIALS
    )

    # Save best hyperparameters and score to a file
    results = {
        "best_params": best_params,
        "best_score": best_score
    }
    results_dir = os.path.join("data", "best_hyperparams", args.dataset, args.strategy, args.model)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "best_hyperparams.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nBusca finalizada!")
    print(f"Melhor F1-macro: {best_score:.4f}")
    print(f"Melhores parâmetros: {best_params}")