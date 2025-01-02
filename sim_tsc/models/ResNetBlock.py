import torch.nn as nn
import torch.nn.functional as F
import lightning as pl

class ResNetBlock(pl.LightningModule):
    """
    Bloco ResNet para redes neurais convolucionais 1D.
    
    Este bloco implementa uma arquitetura residual com três camadas convolucionais,
    normalização em lote (batch normalization) e conexões residuais (skip connections).
    
    Parâmetros:
        in_channels (int): Número de canais de entrada
        out_channels (int): Número de canais de saída
        kernel_size (int, opcional): Tamanho do kernel da convolução. Padrão: 7
        stride (int, opcional): Stride da convolução. Padrão: 1
        padding (int, opcional): Padding da convolução. Padrão: 3
    
    Implementado a partir do código disponível em:
        https://github.com/daochenzha/SimTSC, 
        Daochen Zha, https://doi.org/10.48550/arXiv.2201.01413

    """
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out