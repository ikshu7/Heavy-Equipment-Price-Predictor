# Label Encoding

y = np.log1p(train_df['TargetValue']) # log-transform the skewed target tp match the training to the RMSLE metric, only done on train data 

full = pd.concat([train_df.drop(columns=['TargetValue']), test_df], ignore_index=True) # target value is dropped to prevent leakage, concat is done so that the same codes are applied to both the train and test data

string_cols = [c for c in full.columns if str(full[c].dtype) == 'object']
category_cols = string_cols + ['ProductConfigID']  # productid added seperately because it contains numbers but actualy works as category
# label enoding categorials to codes
for c in category_cols:
    full[c] = pd.Categorical(full[c].astype(str)).codes  

DROP = ['TransactionID', 'TransactionDate'] # transaction id is of no use and already changed the date to yesr/month
# the previously done concat is seperated 
X      = full.iloc[:len(train_df)].drop(columns=DROP).reset_index(drop=True) # drops above columns and resets the index to maintain consistency
X_test = full.iloc[len(train_df):].drop(columns=DROP).reset_index(drop=True)
CAT_FEATURES = [c for c in category_cols if c in X.columns]

print(X.shape, len(CAT_FEATURES), 'categorical features')




# Split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5, labels=False)) # q=5 makes 5 eqaul buckets to maintain same price distribution in train/test




# Target Enc

from sklearn.model_selection import KFold

TE_SMOOTHING = {'AssetID': 2, 'ProductConfigID': 8, 'Spec_FullDescriptor': 8,
                'Spec_BaseClass': 15, 'Spec_SubClass': 20, 'RegionCode': 20,
                'VendorPartnerID': 20, 'FunctionalClassification': 20}

def te_map(k_fit, y_fit, k_apply, prior, m):
    # every ategory mean target and the count of them
    s = pd.DataFrame({'k': k_fit, 'y': y_fit}).groupby('k')['y'].agg(['mean', 'count'])
    enc = (s['mean'] * s['count'] + prior * m) / (s['count'] + m) # if the count is too large then the m value becomes irelevant and actual mean is used, 
                                                                  # if the count is too low then the m dominates and the actual mean becomes closer to global mean
    return pd.Series(k_apply).map(enc).fillna(prior).values # fillna prior fills the global value in new olumns if any

def add_te(X_fit, y_fit, *X_apply):
    X_fit, X_apply = X_fit.copy(), [data.copy() for data in X_apply]
    y_fit = np.asarray(y_fit)
    prior = y_fit.mean() # fallbak/global value
    
    for col, m in TE_SMOOTHING.items():
        k, oof = X_fit[col].values, np.zeros(len(X_fit))
        
        for a, b in KFold(5, shuffle=True, random_state=1).split(X_fit): # divides the data into 5 parts
            oof[b] = te_map(k[a], y_fit[a], k[b], y_fit[a].mean(), m) # trains on 4 parts and apply on the 5th one
        X_fit['te_' + col] = oof
        
        for data in X_apply:
            data['te_' + col] = te_map(k, y_fit, data[col].values, prior, m)
    return (X_fit, *X_apply)

X_train_te, X_val_te = add_te(X_train, y_train, X_val)
print('Target encoding done:', X_train_te.shape)




# SCaling & Pipeline

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

X_train_flat = X_train_te.replace([np.inf, -np.inf], np.nan)
X_val_flat   = X_val_te.replace([np.inf, -np.inf], np.nan)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
    ('scaler', StandardScaler())
])

X_train_scl = pipeline.fit_transform(X_train_flat) # learns mean/std from train
X_val_scl   = pipeline.transform(X_val_flat) # applies mean/std of train

