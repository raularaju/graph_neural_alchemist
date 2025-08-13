from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
import torch
import numpy as np
import pytorch_lightning as pl
from torchmetrics.functional import f1_score


def get_base_args_SAGE_MLPP_4layer(trial, base_args):
    
    nhid = trial.suggest_categorical("nhid", [32, 64, 128])
    
    base_args.h_feats = nhid
    base_args.agg_type = trial.suggest_categorical("agg_type", ["mean", "max", "lstm"])
    return base_args

def get_args_based_on_model(trial, base_args, model_name):
    if model_name == "SAGE_MLPP_4layer":
        return get_base_args_SAGE_MLPP_4layer(trial, base_args)
    else:
        raise ValueError(f"Modelo desconhecido para busca de hiperparâmetros: {model_name}")


def create_objective(dataset, y_labels, model_class_name, base_args):
    
    def objective(trial):
        
        args = get_args_based_on_model(trial, base_args, model_class_name)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in skf.split(range(len(dataset)), y_labels):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

            model = eval(model_class_name)(args)
            trainer = pl.Trainer(
                max_epochs=args.epochs,
                overfit_batches=args.overfit_batches,
                deterministic=True,
                log_every_n_steps = 20,
            )


            trainer.fit(model, train_loader, val_loader)

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    # Adaptar para o formato do batch e forward do modelo
                    # Exemplo genérico para DGL:
                    if isinstance(batch, tuple) or isinstance(batch, list):
                        g, labels = batch
                        preds = model(g).argmax(dim=1)
                    else:
                        # Se dataset retorna outro formato, adaptar aqui
                        raise NotImplementedError("Adapte o batch e forward conforme seu dataset/modelo.")
                    
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

            f1 = f1_score(all_preds, all_labels, average="macro", num_classes=args.num_classes)
            scores.append(f1.item())

        return np.mean(scores)

    return objective