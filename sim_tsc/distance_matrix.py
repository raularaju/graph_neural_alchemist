from dtaidistance import dtw
from scipy.stats import pearsonr
import numpy as np

def create_dtw_matrix(time_series, save_path):
    """
    Cria uma matriz de distâncias DTW (Dynamic Time Warping) entre séries temporais.
    
    Este método:
    1. Verifica se o conjunto de séries temporais é válido
    2. Cria uma matriz vazia para armazenar as distâncias
    3. Calcula a distância DTW entre cada par de séries
    4. Salva a matriz resultante em um arquivo
    
    Parâmetros
    ----------
    time_series : numpy.ndarray
        Array 2D contendo as séries temporais
    save_path : str
        Caminho onde a matriz de distâncias será salva
        
    Retorna
    -------
    numpy.ndarray
        Matriz de distâncias DTW entre as séries temporais
        
    Raises
    ------
    ValueError
        Se o conjunto de séries temporais não for um array 2D
    """
    if(len(np.shape(time_series)) <= 1):
        raise ValueError('The time_series set must be a 2D array with more than one time series')
    
    distance_matrix = np.zeros((len(time_series), len(time_series)))
    
    for i in range(len(time_series)):
        for j in range(i, len(time_series)):
            distance = dtw.distance_fast(
                np.array(time_series[i], dtype=np.double),
                np.array(time_series[j], dtype=np.double))
            distance_matrix[i][j] = distance
    
    np.save(save_path, distance_matrix)    
    return distance_matrix   

def create_pearson_matrix(time_series, save_path):
    """
    Cria uma matriz de correlação de Pearson entre séries temporais.
    
    Este método:
    1. Verifica se o conjunto de séries temporais é válido
    2. Cria uma matriz vazia para armazenar as correlações
    3. Calcula a correlação de Pearson entre cada par de séries
    4. Salva a matriz resultante em um arquivo
    
    Parâmetros
    ----------
    time_series : numpy.ndarray
        Array 2D contendo as séries temporais
    save_path : str
        Caminho onde a matriz de correlações será salva
    """
    if(len(np.shape(time_series)) <= 1):
        raise ValueError('The time_series set must be a 2D array with more than one time series')
    
    covariance_matrix = np.zeros((len(time_series), len(time_series)))
    
    for i in range(len(time_series)):
        for j in range(i, len(time_series)):
            correlation = pearsonr(time_series[i], time_series[j]).statistic
            covariance_matrix[i][j] = correlation
    
    np.save(save_path, covariance_matrix)    
    return covariance_matrix