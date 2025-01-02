import pandas
import numpy as np

def load_usr_dataset_by_name(tsv_file, num_segment, seg_length):    
    data = pandas.read_csv(
        tsv_file, sep="\t", header=None
    )
    
    seg_length = data.shape[1] // num_segment
    
    length= num_segment*seg_length
    
    print(f"data.shape[1]: {data.shape[1]}")
    print(f"length: {length}")
        
    init = data.shape[1] - length
    time_series, labels = data.values[:, init:].astype(np.float64).reshape(
        -1, length, 1
    ), data[0].values.astype(np.int64)    
    lbs = np.unique(labels)
    labels_return = np.copy(labels)
    for idx, val in enumerate(lbs):
        labels_return[labels == val] = idx        
    
    return time_series, labels_return
