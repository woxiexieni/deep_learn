import pandas as pd

def data_loader(string_param):
    data = pd.read_csv(string_param)
    return data
