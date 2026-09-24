import pandas as pd
 
def load_raw_data(data_dir="data/raw/"):
    #Load raw data form the data directory
    train = pd.read_csv(f"{data_dir}train.csv")
    test = pd.read_csv(f"{data_dir}test.csv")
    sample_sub = pd.read_csv(f"{data_dir}sample_sub.csv")