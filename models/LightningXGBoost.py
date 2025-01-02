import pytorch_lightning as pl
import xgboost as xgb
from sklearn.metrics import accuracy_score

class LightningXGBoost(pl.LightningModule):
    """
    Implementa um modelo XGBoost em PyTorch Lightning.
    
    Esta classe implementa um modelo XGBoost que pode ser treinado em um ambiente PyTorch Lightning.
    """
    def __init__(self, args):
        super(LightningXGBoost, self).__init__()
        self.args = args
        # Define o classificador XGBoost
        self.model = xgb.XGBClassifier()
        
    def fit_xgboost(self, X_train, y_train):
        """
        Treina o modelo XGBoost no conjunto de treinamento.
        
        Parâmetros
        ----------
        X_train : array-like
            Conjunto de dados de treinamento
        y_train : array-like
            Rótulos correspondentes aos dados de treinamento
        """
        self.model.fit(X_train, y_train)

    def forward(self, X):
        """
        Realiza a predição usando o modelo XGBoost.
        
        Parâmetros
        ----------
        X : array-like
            Conjunto de dados para predição
        
        Retorna
        -------
        array-like
            Predições do modelo XGBoost
        """
        return self.model.predict(X)

    def training_step(self, batch, batch_idx):
        X, y = batch
        # Converte o lote para arrays numpy para o XGBoost
        X = X.cpu().numpy()
        y = y.cpu().numpy()
        
        # Treina o modelo XGBoost no lote
        self.fit_xgboost(X, y)
        
        # Como o XGBoost lida com a perda internamente, retorna 0
        return 0

    def test_step(self, batch, batch_idx):
        X, y = batch
        X = X.cpu().numpy()
        y = y.cpu().numpy()
        
        # Obtém predições para o conjunto de teste
        preds = self.model.predict(X)
        
        # Calcula a precisão no conjunto de teste
        acc = accuracy_score(y, preds)
        self.log('test_accuracy', acc)

    def configure_optimizers(self):
        # Nenhum otimizador necessário para o XGBoost
        return []