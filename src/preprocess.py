import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

NONE = {'none or unspecified', 'none', 'unspecified', 'nan', 'na', ''}
   
def enigneer_features(df):
    df = df.copy()

    df['ManufactureYear'] = df['ManufactureYear'].where(df['ManufactureYear'] >= 1900, np.nan) # removes the anomaly found in eda

    df['TransactionYear']  = df['TransactionDate'].dt.year
    df['TransactionMonth'] = df['TransactionDate'].dt.month 

    df['AssetAge'] = df['TransactionYear'] - df['ManufactureYear'] # gets the age of the asset/machine
    df.loc[df['AssetAge'] < 0, 'AssetAge'] = np.nan # replaing negative values with nan    

    df['OperationalHoursMeter'] = df['OperationalHoursMeter'].replace(0, np.nan) 
    df['Hours_Per_Year'] = df['OperationalHoursMeter'] / df['AssetAge'].replace(0, np.nan) # assigning nan to avoid getting divided by 0 for new machines
    df['HasHours'] = df['OperationalHoursMeter'].notna().astype(int)

    for c in df.columns:
        if str(df[c].dtype) == 'object':
            mask = df[c].astype(str).str.strip().str.lower().isin(NONE)
            df[c] = df[c].where(~mask, np.nan)

    df['Blank_Column_Count'] = df.isnull().sum(axis=1) # gives empty columns for each row
    df['DescriptorLength']   = df['Spec_FullDescriptor'].astype(str).str.len() # gives length of desription

    df['Model_Popularity'] = df.groupby('ProductConfigID')['TransactionID'].transform('count')
    
    df['Vendor_Volume'] = df.groupby('VendorPartnerID')['TransactionID'].transform('count')
    
    df['Is_Brand_New'] = (df['AssetAge'] == 0).astype(int)
    return df

