import pandas as pd

# ler o arquivo csv

def get_data():
    df = pd.read_csv("data/news.csv")
    df = df[['titulo','texto']]
    return df
