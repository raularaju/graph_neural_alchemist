"""
Módulo de configuração para o framework de Classificação de Séries Temporais.

Este módulo fornece uma configuração centralizada através de argumentos de linha de comando,
abrangendo todos os aspectos do framework:

- Parâmetros de carregamento de dados
- Configurações da arquitetura do modelo
- Hiperparâmetros de treinamento 
- Estratégias de construção de grafos
- Opções de logging e depuração

A configuração é acessível em todo o projeto através da função get_args().
"""

import argparse

def get_args():
    """
    Parse and return command line arguments.
    
    Returns:
        argparse.Namespace: Object containing all configuration parameters
    """
    parser = argparse.ArgumentParser()
    
    # Training Parameters
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Semente aleatória para reprodutibilidade dos experimentos"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Tamanho do lote para treinamento. Valores maiores podem acelerar o treinamento mas consomem mais memória"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001, 
        help="Taxa de aprendizado para o otimizador Adam. Valores típicos entre 0.1 e 0.0001"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0, 
        help="Fator de regularização L2. Ajuda a prevenir overfitting"
    )
    parser.add_argument(
        "--nhid", 
        type=int, 
        default=128, 
        help="Dimensão das camadas ocultas da rede neural"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=250, 
        help="Número máximo de épocas de treinamento"
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=10, 
        help="Número de épocas para aguardar melhoria antes de parar o treinamento (early stopping)"
    )

    # Data Parameters
    parser.add_argument(
        "--train_path", 
        type=str, 
        default="", 
        help="Caminho para o arquivo de treino (formato TSV)"
    )
    parser.add_argument(
        "--test_path", 
        type=str, 
        default="", 
        help="Caminho para o arquivo de teste (formato TSV)"
    )
    parser.add_argument(
        "--tsv_path", 
        type=str, 
        default="", 
        help="Caminho para o arquivo com as séries (formato TSV)"
    )
    parser.add_argument(
        "--dataset", 
        type=str,        
        help="Nome do dataset a ser utilizado. Se não fornecido, será extraído do nome do arquivo de treino"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="/scratch/data",
        help="Diretório raiz para dados de entrada e saída. Os datasets devem estar organizados em subdiretórios"
    )

    # Model Parameters
    parser.add_argument(
        "--model", 
        type=str, 
        default="SAGE_MLPP_4layer", 
        help="Modelo GNN a ser treinado (SAGE_MLPP_4layer, GAT_MLPP, simTSC_GCN, simTSC_SAGE)"
    )
    parser.add_argument(
        "--agg_type", 
        type=str, 
        default="mean", 
        help="Tipo de agregação para GraphSAGE (mean, pool, lstm)"
    )
    parser.add_argument(
        "--dropout",
        type=float, 
        default=0.5,
        help="Taxa de dropout (1 - probabilidade de manter). Usado para regularização"
    )

    # Graph Construction Parameters
    parser.add_argument(
        "--strategy",
        type=str,
        default="vg",
        help="Estratégia de conversão série temporal para grafo (op, vg, simtsc, time2graph, time2graphplus, encoded_gtpo)"
    )
    parser.add_argument(
        "--op_length",
        type=int,
        default=7,
        help="Comprimento do padrão ordinal para estratégias OP e GTPO"
    )
    parser.add_argument(
        "--K",
        type=int,
        default=3,
        help="Número de shapelets para SimTSC e Time2Graph"
    )
    parser.add_argument(
        "--C",
        type=int,
        default=100,
        help="Número de candidatos a shapelet para SimTSC"
    )
    parser.add_argument(
        "--seg_length",
        type=int,
        default=10,
        help="Comprimento dos segmentos para extração de shapelets estáticos"
    )
    parser.add_argument(
        "--num_segment",
        type=int,
        default=10,
        help="Número de segmentos para shapelets estáticos"
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=80,
        help="Percentil para filtragem de similaridade no SimTSC (0-100)"
    )
    parser.add_argument(
        "--R",
        type=float,
        default=0.8,
        help="Limiar R_0 para matriz de correlação de Pearson (0-1)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Peso para combinação de similaridades no SimTSC"
    )
    parser.add_argument(
        "--n_channels",
        type=int,
        default=1,
        help="Número de canais das séries temporais. Use 1 para séries univariadas"
    )

    # Training Control Parameters
    parser.add_argument(
        "--lr_scheduler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Usar scheduler de taxa de aprendizado que reduz quando o treino estagna"
    )
    parser.add_argument(
        "--detect_anomaly",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Detectar anomalias no forward pass do modelo (útil para debugging)"
    )
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Habilitar early stopping baseado na loss de treino"
    )
    parser.add_argument(
        "--overfit_batches",
        type=float,
        default=0.0,
        help="Treinar em uma fração dos batches (útil para debugging)"
    )
    parser.add_argument(
        "--fast_dev_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Executar um treino rápido para teste (apenas alguns batches)"
    )
    parser.add_argument(
        "--fast_dev_run_batches",
        type=int,
        default=4,
        help="Número de batches para executar no modo fast_dev_run"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="lightning_logs",
        help="Diretório para salvar logs do PyTorch Lightning/TensorBoard"
    )

    parser.add_argument(
        "--graphs_test_folder",
        type=str,
        help="Caminho para a pasta dos grafos de teste"
    )
    parser.add_argument(
        "--node_features_test_path",
        type=str,
        help="Caminho para o arquivo de características dos nós de teste"
    )
    parser.add_argument(
        "--graphs_train_folder",
        type=str,
        help="Caminho para a pasta dos grafos de treino"
    )
    parser.add_argument(
        "--node_features_train_path",
        type=str,
        help="Caminho para o arquivo de características dos nós de treino"
    )
    parser.add_argument(
        "--node_features_path",
        type=str,
        help="Caminho para o arquivo de características dos nós"
    )
    parser.add_argument(
        "--graphs_folder",
        type=str,
        help="Caminho para a pasta dos grafos"
    )
    parser.add_argument(
        "--indices_path",
        type=str,
        help="Caminho para índices"
    )



    args = parser.parse_args()
    return args