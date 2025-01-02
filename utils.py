import torch
from scipy.stats import t
import numpy as np
import matplotlib.pyplot as plt
import dgl
import h5py
import pandas as pd
import neurokit2 as nk
from tqdm import tqdm
import wfdb
import os
import networkx as nx
from ts2vg import NaturalVG
from sklearn.preprocessing import OneHotEncoder
import ordpy

import dgl.function as fn
from torch_geometric.data import Data

def get_onehotencoder(train_path, test_path, op_length=3):
    """
    Cria e treina um codificador one-hot para padrões ordinais extraídos de séries temporais.
    
    Esta função:
    1. Carrega dados de treino e teste dos caminhos especificados
    2. Extrai padrões ordinais únicos de todas as séries temporais usando:
        - O método de Bandt-Pompe com comprimento op_length
        - Redes ordinais para mapear padrões
    3. Treina um OneHotEncoder nos padrões únicos encontrados
    
    Args:
        train_path (str): Caminho para o arquivo de dados de treino
        test_path (str): Caminho para o arquivo de dados de teste  
        op_length (int, opcional): Comprimento dos padrões ordinais. Padrão é 3.
        
    Returns:
        OneHotEncoder: Codificador treinado nos padrões ordinais únicos
    """
    
    with open(train_path, 'r') as file:
        train_data = file.readlines()
        
    with open(test_path, 'r') as file:
        test_data = file.readlines()
        
    train_time_series_data = [np.array(data.split("\t")[1:]).astype(np.float32) for data in train_data]
    test_time_series_data = [np.array(data.split("\t")[1:]).astype(np.float32) for data in test_data]
    
    all_time_series_data = train_time_series_data + test_time_series_data
    unique_patterns = set()
    for ts in all_time_series_data:
        nodes, _, _ = ordpy.ordinal_network(ts, op_length)
        unique_patterns.update(nodes)
    unique_patterns = np.array(list(unique_patterns)).reshape(-1, 1)
    
    # Fit the encoder on the unique patterns from both training and testing data
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(unique_patterns)    
    return encoder

def collate_fn(batch):
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)

    batch_tensor = []
    for i, g in enumerate(graphs):
        batch_tensor.extend([i] * g.number_of_nodes())
    batch_tensor = torch.tensor(batch_tensor, dtype=torch.long)

    return batched_graph, labels, batch_tensor

def dgl_to_pyg(dgl_graph, batch):
    edge_index = torch.stack(dgl_graph.edges())
    x = dgl_graph.ndata['feat']
    edge_weight = dgl_graph.edata['weight']    
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, batch=batch)    
    return pyg_data

def compute_pagerank(graph, N=100, K=10, DAMP=0.85):
    """
    Calcula o PageRank para cada nó do grafo usando o algoritmo iterativo de Power Method.
    
    Este método:
    1. Inicializa um vetor de probabilidade uniforme para todos os nós
    2. Para cada iteração K:
        - Normaliza o vetor pelo grau de saída dos nós
        - Propaga as probabilidades através das arestas do grafo
        - Aplica o fator de amortecimento (damping factor)
    3. Retorna o vetor final de PageRank
    
    Args:
        graph: Grafo DGL de entrada
        N (int): Número de nós no grafo (padrão: 100)
        K (int): Número de iterações do algoritmo (padrão: 10) 
        DAMP (float): Fator de amortecimento entre 0 e 1 (padrão: 0.85)
    
    Returns:
        torch.Tensor: Vetor com os valores de PageRank para cada nó
    """
    graph.ndata["pv"] = torch.ones(N) / N
    degrees = graph.out_degrees(graph.nodes()).type(torch.float32)
    for _ in range(K):
        graph.ndata["pv"] = graph.ndata["pv"] / degrees
        graph.update_all(
            message_func=fn.copy_u(u="pv", out="m"),
            reduce_func=fn.sum(msg="m", out="pv"),
        )
        graph.ndata["pv"] = (1 - DAMP) / N + DAMP * graph.ndata["pv"]
    return graph.ndata["pv"]
''' 
    Process MIT-BIH Arrhythmia Database based on the paper by Oliveira et al. Explorando Redes Neurais de Grafos para Classificacão de Arritmias
    https://sol.sbc.org.br/index.php/sbcas/article/view/21630
'''

def processa_mit_bih(registros, tsv_file):        
    for registro in tqdm(registros, desc=f"Processando registros..."):
        record = wfdb.rdrecord(registro, pn_dir='mitdb')
        annotation = wfdb.rdann(registro, 'atr', pn_dir='mitdb')

        # Extract Heartbeats with Labels
        signal = record.p_signal[:, 0]  # Assuming a single lead ECG signal, derivation MLII
        
        window_size = 150
        
        for i, peak in enumerate(annotation.sample):
            start_index = peak - window_size
            end_index = peak + window_size            
            if start_index >= 0 and end_index < len(signal):
                time_series = []
                label = annotation.symbol[i]

                # group labels into 5 classes based on AAMI standard
                if label in ['N', 'L', 'R', 'e', 'j']:
                    label = 'N'
                elif label in ['A', 'a', 'J', 'S']:
                    label = 'S'
                elif label in ['V', 'E']:
                    label = 'V'
                elif label in ['F']:
                    label = 'F'
                elif label in ['/', 'f', 'Q']:
                    label = 'Q'
                else:
                    label = 'U' # unknown

                # save only the labels of interest, N, S, V
                
                if label in ['N', 'S', 'V']:
                    
                    #replace the labels by numbers
                    if label == 'N':
                        label = 0
                    elif label == 'S':
                        label = 1
                    elif label == 'V':
                        label = 2
                    
                    time_series.append(registro)
                    time_series.append(label) # append the label to the list
                    time_series.extend(signal[start_index:end_index]) # extend the signal to the list                                       
                    
                    #if path does not exist, create it
                    if not os.path.exists(os.path.dirname(tsv_file)):
                        os.makedirs(os.path.dirname(tsv_file))
                    
                    with open(tsv_file, 'a+') as file:                    
                        file.write("\t".join(map(str, time_series)) + "\n")

def processa_CODE15(hdf_data_path, labels_path, tsv_file, nro_samples=2000, fixed_size=False):
    print("Processing CODE15 dataset")
    
    with h5py.File(hdf_data_path, 'r') as hdf5_file:
        tracings = np.array(hdf5_file["tracings"][:nro_samples])
        exam_ids = np.array(hdf5_file["exam_id"][:nro_samples])
    
    print("processing the labels and filtering the tracings")
    
    labels_df = pd.read_csv(labels_path, skipinitialspace=True, usecols=['exam_id', 'label'])
    labels_df["label"].replace("AF", 0, inplace=True)
    labels_df["label"].replace("SB", 1, inplace=True)
    labels_df["label"].replace("RBBB", 2, inplace=True)
    labels_df["label"].replace("1dAVb", 3, inplace=True)
    labels_df["label"].replace("ST", 4, inplace=True)
    labels_df["label"].replace("LBBB", 5, inplace=True)
    
    tracings = tracings[np.isin(exam_ids, labels_df['exam_id'].values)]
    exam_ids = np.intersect1d(exam_ids, labels_df['exam_id'].values)

    for trace, exam_id in tqdm(zip(tracings, exam_ids), desc=f"Processing traces, fixed size == {fixed_size}", total=len(exam_ids)):        
        label = labels_df[labels_df['exam_id'] == exam_id]['label'].values[0]
        signal = trace.T[1] # get only the DII signal derivation

        # process the signal with neurokit and segment the signal in heartbeats

        ecg, info = nk.ecg_process(signal, sampling_rate=400, method="neurokit") 
        
        if(not fixed_size):
            heartbeats = nk.ecg_segment(ecg_cleaned=ecg["ECG_Clean"], rpeaks=info["ECG_R_Peaks"], sampling_rate=400, show=False)
            heartbeats = nk.epochs_to_df(heartbeats)

            #Get only the heartbeats with signal different from 0 and Nan
            cleaned_heartbeats = heartbeats[(heartbeats['Signal'] != 0) & (~heartbeats['Signal'].isna())]

            #get the signal of each Label and store in a numpy array
            for hearbeat_nro in cleaned_heartbeats['Label'].unique():            
                heartbeat_signal = cleaned_heartbeats[cleaned_heartbeats['Label'] == hearbeat_nro]['Signal'].values
                
                time_series = []
                time_series.append(exam_id)
                time_series.append(label)
                time_series.extend(heartbeat_signal)
                with open(tsv_file, 'a') as file:
                    file.write("\t".join(map(str, time_series)) + "\n")
        
        else:
            window_size = 150
            if(len(info['ECG_R_Peaks']) > 0):
                for peak in info['ECG_R_Peaks']:
                    start_index = peak - window_size
                    end_index = peak + window_size            
                    if start_index >= 0 and end_index < len(signal):
                        time_series = []            
                        time_series.append(exam_id)
                        time_series.append(label) # append the label to the list
                        time_series.extend(signal[start_index:end_index]) # extend the signal to the list                                       
                        with open(tsv_file, 'a+') as file:                    
                            file.write("\t".join(map(str, time_series)) + "\n")

def plota_distribuicao_classes(dataset, x_labels=None, plot=True):
    
    # Get all labels from the dataset
    dataset_labels = dataset.labels
    
    class_counts = np.bincount(dataset_labels)
    num_classes = len(class_counts)
    class_indices = np.arange(num_classes)

    class_ratios = np.round(class_counts / len(dataset_labels), 2)

    if plot:
        fig, ax = plt.subplots()
        
        # Bar plot with modern color scheme
        bars = ax.bar(class_indices, class_counts, color="#4CAF50", label="Class Distribution")
        
        # Plot normal distribution (if needed)
        # ax.plot(class_indices, norm.pdf(class_indices, np.mean(class_indices), np.std(class_indices)),
        #         color="red", label="Normal Distribution")
        
        # X-axis and Y-axis labels
        ax.set_xlabel("Class Label" if x_labels is None else "Class")
        ax.set_ylabel("Frequency")
        ax.set_title("Class Distribution")
        
        # Set custom x-axis labels if provided
        if x_labels:
            ax.set_xticks(class_indices)
            ax.set_xticklabels(x_labels)
        
        # Legend
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        
        # Grid
        ax.grid(axis="y")
        
        # Incorporate class_counts and class_ratios into the plot
        for bar, count, ratio in zip(bars, class_counts, class_ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f"{count}", 
                    ha="center", va="bottom", color="#2196F3", fontweight="bold")
            
            # Center class ratios in the middle of the bars
            ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f"{ratio}", 
                    ha="center", va="center", color="#FFFFFF", fontweight="bold")
        plt.show()

def plota_grafo_visibilidade(tsv_file, dict_labels):
    with open(tsv_file, 'r') as file:
        train_data = file.readlines()

    #get number of labels of the dataset
    nro_labels = len(set([int(data[0]) for data in train_data]))
    labels = []
    labels_processed = 0
    #get one sample per label and plot the visibility graph
    for data in train_data:

        # 1. Get the label and the signal
        label = int(data[0])
        signal = np.array(data.split("\t")[1:]).astype(np.float32)
        
        if(labels_processed == nro_labels):
            break
        
        #check if label was not processed yet
        if label not in labels:
            labels.append(label)
            labels_processed += 1

            # 2. Build the visibility graph
            vis_graph = NaturalVG(directed="left_to_right", weighted="sq_distance").build(signal)
            nxg = vis_graph.as_networkx()
            
            # 3. Make plots
            fig, [ax0, ax1] = plt.subplots(ncols=2, figsize=(12, 3.5))

            #set title for the figure
            fig.suptitle(f"Plots para o Label: {dict_labels[label]}")

            #put the number of nodes and edges in the figure
            fig.text(0.05, 0.95, f"Nro de nós: {len(nxg.nodes)}", ha='right', va='top', fontsize=9)
            fig.text(0.05, 0.9, f"Nro de arestas: {len(nxg.edges)}", ha='right', va='top', fontsize=9)

            #make space between the text and the plots
            # fig.subplots_adjust(top=0.75)

            ax0.plot(vis_graph.ts)
            ax0.set_title("Time Series")

            graph_plot_options = {
                "with_labels": False,
                "node_size": 2,
                "node_color": [(0, 0, 0, 1)],
                "edge_color": [(0, 0, 0, 0.15)],
            }

            nx.draw_networkx(nxg, ax=ax1, pos=vis_graph.node_positions(), **graph_plot_options)
            ax1.tick_params(bottom=True, labelbottom=True)
            ax1.plot(vis_graph.ts)
            ax1.set_title("Visibility Graph")

def set_seeds(model, train_loader, eval_loader, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    np.random.seed(seed)
    dgl.random.seed(seed)
    dgl.seed(seed)    
    model.seed = seed
    train_loader.seed = seed
    eval_loader.seed = seed