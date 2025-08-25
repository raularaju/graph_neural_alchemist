import torch
from torch.nn import functional as F
import lightning.pytorch as pl
import seaborn as sn
import io
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import os

from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    confusion_matrix,
)
from sim_tsc import run_simTSC
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LightningGNN(pl.LightningModule):
    """
    Módulo Lightning para treinamento e avaliação de Redes Neurais baseadas em Grafos (GNNs).
    
    Esta classe implementa a lógica de treinamento, validação e teste para modelos GNN,
    incluindo métricas de avaliação, logging e visualizações.

    Atributos:
        seed (int): Semente para reprodutibilidade
        args (argparse.Namespace): Argumentos de configuração
        model (torch.nn.Module): Modelo GNN a ser treinado
        training_metrics (dict): Métricas coletadas durante treinamento
        validation_metrics (dict): Métricas coletadas durante validação 
        test_metrics (dict): Métricas coletadas durante teste

    Métodos:
        __fit__: Processa um lote de dados e retorna predições
        loss_function: Calcula a função de perda
        configure_optimizers: Configura otimizador e scheduler
        training_step: Executa uma etapa de treinamento
        validation_step: Executa uma etapa de validação
        test_step: Executa uma etapa de teste
    """

    def __init__(self, args, model, save_path = None):
        """
        Inicializa o módulo Lightning GNN.

        Args:
            args (argparse.Namespace): Argumentos de configuração
            model (torch.nn.Module): Modelo GNN a ser treinado
        """
        super(LightningGNN, self).__init__()
        self.seed = args.seed
        self.args = args
        self.model = model

        self.training_metrics = {"true": [], "loss": []}
        self.validation_metrics = {"pred": [], "true": [], "loss": []}
        self.test_metrics = {"pred": [], "true": [], "loss": []}
        if save_path is None:
            self.save_path = os.path.join("lightning_logs", self.args.save_dir, self.args.dataset, self.args.model)
        else:
            self.save_path = save_path
        args.model_architecture = self.model.__str__().replace("\n", "")
        self.save_hyperparameters(args)    
        
    def __fit__(self, step_batch, batch_idx, type_step):
        """
        Processa um lote de dados e retorna predições.

        Args:
            step_batch (tuple): Tupla contendo grafo e rótulos
            batch_idx (int): Índice do lote atual
            type_step (str): Tipo de etapa ('training', 'validating' ou 'testing')

        Returns:
            tuple: Rótulos e logits preditos
        """
        graph, labels = step_batch        
        
        ########## bandt_pompe or Ordinal (op) strategy ##########
        if self.args.strategy == "op":            
            input_data = graph.ndata["feat"]
            logits = self.model(graph, input_data, graph.edata['weight'])
        
        ########## SimTSC and Pearson strategies ##########
        ## Aqui o input_data é um grafo DGL de simTSC, e graph é a série temporal original
        
        elif self.args.strategy in ["simtsc"]:
            input_data = run_simTSC.run(batch_idx, labels, graph, self.args, type_step)
            logits = self.model(graph, input_data)      
        
        else:
            input_data = graph.ndata["feat"]
            logits = self.model(graph, input_data)
        
        return labels,logits          

    def loss_function(self, logits, labels):
        """
        Calcula a função de perda cross-entropy.

        Args:
            logits (torch.Tensor): Predições do modelo
            labels (torch.Tensor): Rótulos verdadeiros

        Returns:
            torch.Tensor: Valor da perda
        """
        return F.cross_entropy(logits, labels)

    def __cria_matriz_confusao(self, y_pred, y_true):
        """
        Cria e retorna uma matriz de confusão como imagem.

        Args:
            y_pred (torch.Tensor): Predições do modelo
            y_true (torch.Tensor): Rótulos verdadeiros

        Returns:
            torch.Tensor: Imagem da matriz de confusão
        """
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        _, ax = plt.subplots(figsize=(10, 5))

        sn.set_theme(font_scale=1.2)
        sn.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt="d", ax=ax)
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")

        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg", bbox_inches="tight")
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        return im

    def __custom_histogram_adder(self):
        """
        Adiciona histogramas dos parâmetros do modelo ao logger.
        """
        for name, param in self.model.named_parameters():
            self.logger.experiment.add_histogram(name, param, self.current_epoch)

    def configure_optimizers(self):
        """
        Configura otimizador e scheduler de taxa de aprendizado.

        Returns:
            dict: Configuração do otimizador e scheduler
        """        
        if(self.args.weight_decay > 0):
            optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            print(f"====> Using weight decay of {self.args.weight_decay} <====")
        else:
            optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        
        optimizer.seed = self.args.seed

        if self.args.lr_scheduler:
            factor = 0.1
            patience = self.args.patience
            lr_scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=factor, patience=patience, verbose=True
            )

            print(
                f"====> Using ReduceLROnPlateau to adapt the learning rate to min with factor {factor} and patience {patience} <===="
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": "train_loss",
            }

        print(":) Not using ReduceLROnPlateau to adapt the learning rate :)")
        return {"optimizer": optimizer, "monitor": "train_loss"}

    def training_step(self, train_batch, batch_idx):
        """
        Executa uma etapa de treinamento.

        Args:
            train_batch (tuple): Lote de dados de treinamento
            batch_idx (int): Índice do lote

        Returns:
            dict: Dicionário contendo a perda
        """        
        labels, logits = self.__fit__(train_batch, batch_idx, 'training')
        train_loss = self.loss_function(logits, labels)
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=labels.size(0),
        )
        self.logger.experiment.add_scalar("train_loss", train_loss, self.current_epoch)
        self.training_metrics["loss"].append(train_loss)

        return {"loss": train_loss}

    def on_train_epoch_end(self) -> None:
        """
        Executa operações ao final de cada época de treinamento.
        """
        avg_train_loss = torch.stack(self.training_metrics["loss"]).mean()
        self.log(
            "avg_train_loss",
            avg_train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        self.logger.experiment.add_scalar(
            "avg_train_loss", avg_train_loss, self.current_epoch
        )
        self.__custom_histogram_adder()
        self.training_metrics = {"true": [], "loss": []}        

    def validation_step(self, val_batch, batch_idx):
        """
        Executa uma etapa de validação.

        Args:
            val_batch (tuple): Lote de dados de validação
            batch_idx (int): Índice do lote

        Returns:
            dict: Dicionário contendo métricas de validação
        """
        labels, logits = self.__fit__(val_batch, batch_idx, 'validating')

        val_loss = self.loss_function(logits, labels)
        y_pred = logits.argmax(dim=1, keepdim=True).flatten()

        self.validation_metrics["pred"].extend(y_pred)
        self.validation_metrics["true"].extend(labels)

        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=labels.size(0),
        )

        precision = precision_score(y_pred, labels, average="macro",zero_division=0)
        recall = recall_score(y_pred, labels, average="macro", zero_division=0)
        f1 = f1_score(y_pred, labels, average="macro")
        mcc = matthews_corrcoef(y_pred, labels)
        acc = accuracy_score(y_pred, labels)

        self.log(
            "val_precision",
            precision,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val_recall",
            recall,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val_f1_score",
            f1,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val_mcc", mcc, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        return {"loss": val_loss, "pred": y_pred, "true": labels, "total": len(labels)}

    def on_validation_epoch_end(self) -> None:
        """
        Executa operações ao final de cada época de validação.
        """
        y_pred = torch.stack(self.validation_metrics["pred"])
        y_true = torch.stack(self.validation_metrics["true"])

        val_confusion_matrix = self.__cria_matriz_confusao(y_pred=y_pred, y_true=y_true)
        self.logger.experiment.add_image(
            "val_confusion_matrix", val_confusion_matrix, self.current_epoch
        )
        
        self.validation_metrics = {"pred": [], "true": [], "loss": []}

    def test_step(self, test_batch, batch_idx):
        """
        Executa uma etapa de teste.

        Args:
            test_batch (tuple): Lote de dados de teste
            batch_idx (int): Índice do lote

        Returns:
            dict: Dicionário contendo métricas de teste
        """
        labels, logits = self.__fit__(test_batch, batch_idx, 'testing')
        test_loss = self.loss_function(logits, labels)
        y_pred = logits.argmax(dim=1, keepdim=True).flatten()

        self.test_metrics["pred"].extend(y_pred)
        self.test_metrics["true"].extend(labels)

        self.log(
            "test_loss",
            test_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=labels.size(0),
        )

        return {"loss": test_loss, "pred": y_pred, "y": labels, "total": len(labels)}

    def on_test_epoch_end(self) -> None:
        """
        Executa operações ao final de cada época de teste.
        Salva predições, gera relatórios e registra métricas finais.
        """
        y_pred = torch.stack(self.test_metrics["pred"]).cpu().numpy()
        y_true = torch.stack(self.test_metrics["true"]).cpu().numpy()
        
        #save the predictions and true labels for further analysis
        
        os.makedirs(self.save_path, exist_ok=True)
        with open(f"{self.save_path}/predictions_best_model.csv", "w") as file:
            file.write("pred,true\n")
            for i in range(len(y_pred)):
                file.write(f"{y_pred[i]},{y_true[i]}\n")
            print(f"====> Predictions saved to {self.save_path}/predictions_best_model.csv <====")
            
        test_confusion_matrix = self.__cria_matriz_confusao(
            y_pred=y_pred, y_true=y_true
        )
        self.logger.experiment.add_image(
            "test_confusion_matrix", test_confusion_matrix, self.current_epoch
        )

        print(f"====> Classification Report <====")
        class_report = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0)
        print(class_report)

        precision = precision_score(y_pred, y_true, average="macro",zero_division=0)
        recall = recall_score(y_pred, y_true, average="macro", zero_division=0)
        f1 = f1_score(y_pred, y_true, average="macro")
        mcc = matthews_corrcoef(y_pred, y_true)
        acc = accuracy_score(y_pred, y_true)
        
        print(f"Dataset: {self.args.dataset}")
        
        self.log(
            "test_precision",
            precision,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "test_recall",
            recall,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "test_f1_score",
            f1,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "test_mcc", mcc, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            "test_acc", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )        

        # pretty format class_report before adding to tensorboard
        class_report = class_report.replace("\n", "<br>")
        class_report = class_report.replace(" ", "&nbsp;")
        class_report = class_report.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")

        self.logger.experiment.add_text(
            "test_classification_report_sklearn", class_report, self.current_epoch
        )
        
        self.test_metrics = {"pred": [], "true": [], "loss": []}