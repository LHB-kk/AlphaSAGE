import numpy as np
import os
from alphagen.data.expression import *
from alphagen.utils import reseed_everything
from alphagen_generic.features import *
from gan.utils.data import get_data_by_year

# # Utility functions

import pandas as pd
from tqdm import tqdm
def get_ml_data(data):
    df = data.df_bak.copy()
    print(df.shape)
    print(df.columns)
    print("=" * 20)
    df.columns = ['open','close','high','low','volume', "vwap"]
    close_unstack = df['close'].unstack()
    tmp = (close_unstack.shift(-20)/close_unstack)-1
    label = tmp.stack().reindex(df.index)
    df['label'] = label

    feature = df[['open','close','high','low','volume',]]
    tmp = feature.unstack()
    feature = (tmp/tmp.shift(1)-1)#.stack().reindex(df.index)

    result_feature = []
    cur = feature.stack().reindex(df.index)
    cur.columns = [f'{col}0' for col in cur.columns]
    for past in tqdm(range(1,60)):
        cur = feature.shift(past).stack().reindex(df.index)
        cur.columns = [f'{col}{past}' for col in cur.columns]
        result_feature.append(cur)
    result_feature = pd.concat(result_feature,axis=1)
    df = pd.concat([result_feature,df['label']],axis=1)
    start_date = data._dates[data.max_backtrack_days]
    end_date = data._dates[-data.max_future_days]
    return df.loc[start_date:end_date]

def normalize_data(df_train, df_valid, df_test):
    # Get the column names of the features
    labels = [df_train.iloc[:, -1], df_valid.iloc[:, -1], df_test.iloc[:, -1]]
    df_train_features = df_train.iloc[:, :-1]
    df_valid_features = df_valid.iloc[:, :-1]
    df_test_features = df_test.iloc[:, :-1]

    _mean = df_train_features.mean()
    _std = df_train_features.std()
    print('1')
    df_train_norm = (df_train_features - _mean) / _std
    print('2')
    df_valid_norm = (df_valid_features - _mean) / _std
    print('3')
    df_test_norm = (df_test_features - _mean) / _std

    df_train_norm.fillna(0, inplace=True)
    df_valid_norm.fillna(0, inplace=True)
    df_test_norm.fillna(0, inplace=True)


    df_train_norm['label'] = np.nan_to_num(labels[0],nan=0,posinf=0,neginf=0)
    df_valid_norm['label'] = np.nan_to_num(labels[1],nan=0,posinf=0,neginf=0)
    df_test_norm['label'] = np.nan_to_num(labels[2],nan=0,posinf=0,neginf=0)

    df_train_norm['label'] = df_train_norm['label'].groupby('datetime').transform(lambda x: (x - x.mean()) / x.std()).clip(-4, 4)

    df_train_norm = df_train_norm.clip(-4, 4)
    df_valid_norm = df_valid_norm.clip(-4, 4)
    df_test_norm = df_test_norm.clip(-4, 4)

    return df_train_norm, df_valid_norm, df_test_norm

# # Train LightGBM Model


import lightgbm as lgb
import pandas as pd

def train_lightgbm_model(df_train, df_valid, df_test):
    # Fill NaN values with 0
    df_train_filled = df_train.fillna(0)
    df_valid_filled = df_valid.fillna(0)
    df_test_filled = df_test.fillna(0)

    # Separate features and labels
    X_train = df_train_filled.drop(columns=['label'])
    y_train = df_train_filled['label']
    X_valid = df_valid_filled.drop(columns=['label'])
    y_valid = df_valid_filled['label']
    X_test = df_test_filled.drop(columns=['label'])
    y_test = df_test_filled['label']

    # Convert data to LightGBM Dataset format
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    # Set hyperparameters for LightGBM model
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 128,
        'max_depth': 64,
        'learning_rate': 0.2,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    # Train the LightGBM model
    model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], num_boost_round=100, callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    
    pred = pd.concat([pd.Series(y_pred,index=df_test.index),df_test['label']],axis=1)
    # Print the RMSE score
    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # print(f"RMSE: {rmse}")

    return  model,pred

for instruments in ['sp500']:
    for train_end in range(2016,2019):
        returned = get_data_by_year(
            train_start = 2010, train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
            instruments=instruments, target=target,freq='day',
            qlib_path='/root/autodl-tmp/qlib_data/us_data'
            )
        data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,_ = returned
        df_train = get_ml_data(data)
        df_valid = get_ml_data(data_valid)
        df_test = get_ml_data(data_test)
        df_train, df_valid, df_test = normalize_data(df_train, df_valid, df_test)
        
        model_name = 'lgbm'
        name = f"{instruments}_{model_name}_{train_end}"
        os.makedirs(f"out_ml/{name}",exist_ok=True)
        model,pred = train_lightgbm_model(df_train, df_valid, df_test)
        model.save_model(f"out_ml/{name}/{model_name}.pt")
        pred.to_pickle(f"out_ml/{name}/pred.pkl")


# # Train XGBoost Model



import xgboost as xgb
import pandas as pd

def train_xgboost_model(df_train, df_valid, df_test):
    # Fill NaN values with 0
    df_train_filled = df_train.fillna(0)
    df_valid_filled = df_valid.fillna(0)
    df_test_filled = df_test.fillna(0)

    # Separate features and labels
    X_train = df_train_filled.drop(columns=['label'])
    y_train = df_train_filled['label']
    X_valid = df_valid_filled.drop(columns=['label'])
    y_valid = df_valid_filled['label']
    X_test = df_test_filled.drop(columns=['label'])
    y_test = df_test_filled['label']

    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # Set hyperparameters for XGBoost model
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'colsample_bytree': 0.8879,
        'learning_rate': 0.01,
        'subsample': 0.8789,
        'lambda': 205.6999,
        'alpha': 580.9768,
        'max_depth': 4,
        'num_boost_round': 100,
        'early_stopping_rounds': 100,
        'verbose_eval': 100
    }

    # Train the XGBoost model
    model = xgb.train(params, dtrain, evals=[(dtrain, 'train'), (dvalid, 'valid')], early_stopping_rounds=params['early_stopping_rounds'], verbose_eval=params['verbose_eval'])

    # Convert test data to DMatrix format
    dtest = xgb.DMatrix(X_test)

    # Make predictions on the test set
    y_pred = model.predict(dtest)

    # Combine the predictions with the actual labels
    # pred = pd.concat([df_test['label'], pd.Series(y_pred, index=df_test.index)], axis=1)
    pred = pd.concat([pd.Series(y_pred,index=df_test.index),df_test['label']],axis=1)

    return model, pred



for instruments in ['sp500']:
    for train_end in range(2016,2019):
        returned = get_data_by_year(
            train_start = 2010,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
            instruments=instruments, target=target,freq='day',
            qlib_path='/root/autodl-tmp/qlib_data/us_data'
            )
        data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,_ = returned
        df_train = get_ml_data(data)
        df_valid = get_ml_data(data_valid)
        df_test = get_ml_data(data_test)
        df_train, df_valid, df_test = normalize_data(df_train, df_valid, df_test)
        
        model_name = 'xgb'
        name = f"{instruments}_{model_name}_{train_end}"
        os.makedirs(f"out_ml/{name}",exist_ok=True)
        model,pred = train_xgboost_model(df_train, df_valid, df_test)
        model.save_model(f"out_ml/{name}/{model_name}.pt")
        pred.to_pickle(f"out_ml/{name}/pred.pkl")


# # Train MLP Model


import torch
import torch.nn as nn
import torch.optim as optim

def train_mlp_model(df_train, df_valid, df_test):
    # Fill NaN values with 0
    df_train_filled = df_train.fillna(0)
    df_valid_filled = df_valid.fillna(0)
    df_test_filled = df_test.fillna(0)

    # Separate features and labels
    X_train = df_train_filled.drop(columns=['label']).values
    y_train = df_train_filled['label'].values
    X_valid = df_valid_filled.drop(columns=['label']).values
    y_valid = df_valid_filled['label'].values
    X_test = df_test_filled.drop(columns=['label']).values
    y_test = df_test_filled['label'].values

    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Define the MLP model
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)


    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the model
    num_epochs = 2
    batch_size = 512
    for epoch in range(num_epochs):
        # Shuffle the training data
        indices = torch.randperm(X_train.shape[0])
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # Mini-batch training
        for i in tqdm(range(0, X_train.shape[0], batch_size)):
            # Get the mini-batch
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            # Move the mini-batch to GPU if available
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs.flatten(), y_batch.flatten())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate the model on the test set
    with torch.no_grad():
        # Move the test data to GPU if available
        test_outputs = model(X_test.to(device)).detach().cpu().numpy().flatten()
        pred_df = pd.concat([df_test['label'],pd.Series(test_outputs,index=df_test.index)],axis=1)
    torch.cuda.empty_cache()
    return model, pred_df

for instruments in ['sp500']:
    for train_end in range(2016,2019):
        returned = get_data_by_year(
            train_start = 2010,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
            instruments=instruments, target=target,freq='day',)
        data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,_ = returned
        df_train = get_ml_data(data)
        df_valid = get_ml_data(data_valid)
        df_test = get_ml_data(data_test)
        df_train, df_valid, df_test = normalize_data(df_train, df_valid, df_test)
        for seed in range(1):
            reseed_everything(seed)
            model_name = 'mlp'
            name = f"{instruments}_{model_name}_{train_end}"
            os.makedirs(f"out_ml/{name}",exist_ok=True)
            model,pred = train_mlp_model(df_train, df_valid, df_test)
            # model.save_model(f"out_ml/{name}/model.pt")
            pred.to_pickle(f"out_ml/{name}/pred_{seed}.pkl")


# # Show LightGBM Resuls


import pandas as pd
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr, batch_ret, batch_sharpe_ratio, batch_max_drawdown
import torch
import os
import numpy as np

def chunk_batch_spearmanr(x, y, chunk_size=100):
    n_days = len(x)
    spearmanr_list= []
    for i in range(0, n_days, chunk_size):
        spearmanr_list.append(batch_spearmanr(x[i:i+chunk_size], y[i:i+chunk_size]))
    spearmanr_list = torch.cat(spearmanr_list, dim=0)
    return spearmanr_list

def get_tensor_metrics(x, y, risk_free_rate=0.0):
    # Ensure tensors are 2D (days, stocks)
    if x.dim() > 2: x = x.squeeze(-1)
    if y.dim() > 2: y = y.squeeze(-1)

    ic_s = batch_pearsonr(x, y)
    ric_s = chunk_batch_spearmanr(x, y, chunk_size=400)
    ret_s = batch_ret(x, y) - 0.001

    ic_s = torch.nan_to_num(ic_s, nan=0.)
    ric_s = torch.nan_to_num(ric_s, nan=0.)
    ret_s = torch.nan_to_num(ret_s, nan=0.) / 20
    ic_s_mean = ic_s.mean().item()
    ic_s_std = ic_s.std().item() if ic_s.std().item() > 1e-6 else 1.0
    ric_s_mean = ric_s.mean().item()
    ric_s_std = ric_s.std().item() if ric_s.std().item() > 1e-6 else 1.0
    ret_s_mean = ret_s.mean().item()
    ret_s_std = ret_s.std().item() if ret_s.std().item() > 1e-6 else 1.0
    
    # Calculate Sharpe Ratio and Maximum Drawdown for ret series
    ret_sharpe = batch_sharpe_ratio(ret_s, risk_free_rate).item()
    ret_mdd = batch_max_drawdown(ret_s).item()
    result = dict(
        ic=ic_s_mean,
        ic_std=ic_s_std,
        icir=ic_s_mean / ic_s_std,
        ric=ric_s_mean,
        ric_std=ric_s_std,
        ricir=ric_s_mean / ric_s_std,
        ret=ret_s_mean * len(ret_s) / 3,
        ret_std=ret_s_std,
        retir=ret_s_mean / ret_s_std,
        ret_sharpe=ret_sharpe,
        ret_mdd=ret_mdd,
    )
    return result, ret_s




instruments = 'sp500'

result = []
for year in range(2016,2019):
    result.append(pd.read_pickle(f'out_ml/{instruments}_lgbm_{year}/pred.pkl'))
df = pd.concat(result,axis=0)
data = df.pivot_table(index="datetime", columns="instrument", values=[0,"label"])
pred = data[0].values
label = data["label"].values
res, ret_s = get_tensor_metrics(torch.tensor(pred), torch.tensor(label))
print(pd.DataFrame(res,index=['Test']))
save_path = os.path.join(f'out_ml/{instruments}_lgbm_{year}', 'ret_s.npy')
np.save(save_path, ret_s)


# # Show XGBoost Result


import pandas as pd
instruments = 'sp500'

result = []
for year in range(2016,2019):
    result.append(pd.read_pickle(f'out_ml/{instruments}_xgb_{year}/pred.pkl'))
df = pd.concat(result,axis=0)

data = df.pivot_table(index="datetime", columns="instrument", values=[0,"label"])
pred = data[0].values
label = data["label"].values
res, ret_s = get_tensor_metrics(torch.tensor(pred), torch.tensor(label))
print(pd.DataFrame(res,index=['Test']))
save_path = os.path.join(f'out_ml/{instruments}_xgb_{year}', 'ret_s.npy')
np.save(save_path, ret_s)


# # Show MLP Result


import pandas as pd
result_all = []
instruments = 'sp500'
for seed in range(1):
    result = []
    for year in range(2016,2019):
        result.append(pd.read_pickle(f'out_ml/{instruments}_mlp_{year}/pred_{seed}.pkl'))
    df = pd.concat(result,axis=0)#.groupby('datetime').corr('spearman')['label'].unstack().mean()
    data = df.pivot_table(index="datetime", columns="instrument", values=[0,"label"])
    pred = data[0].values
    label = data["label"].values
    res, ret_s = get_tensor_metrics(torch.tensor(pred), torch.tensor(label))
    print(pd.DataFrame(res,index=['Test']))
    save_path = os.path.join(f'out_ml/{instruments}_mlp_{year}', 'ret_s.npy')
    np.save(save_path, ret_s)





