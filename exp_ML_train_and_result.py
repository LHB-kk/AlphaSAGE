import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import lightgbm as lgb
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim

from alphagen.utils import reseed_everything
from alphagen_generic.features import target
from gan.utils.data import get_data_by_year

# # Utility functions
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr, batch_ret, batch_sharpe_ratio, batch_max_drawdown

def get_ml_data(data):
    df = data.df_bak.copy()
    print(df.shape)
    print(df.columns)
    print("=" * 20)
    df.columns = ['open','close','high','low','volume', "vwap"]
    # 构造标签
    close_unstack = df['close'].unstack()
    tmp = (close_unstack.shift(-20)/close_unstack)-1 # 未来20天的收益率
    label = tmp.stack().reindex(df.index)
    df['label'] = label

    # 对每个原始价格/成交量/VWAP计算日收益率
    feature = df[['open','close','high','low','volume', "vwap"]]
    tmp = feature.unstack()
    feature = (tmp/tmp.shift(1)-1)

    # 构造特征（60天历史收益率）
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
    # 标准化特征、归一化特征，做Z-score归一化
    labels = [df_train.iloc[:, -1], df_valid.iloc[:, -1], df_test.iloc[:, -1]]
    df_train_features = df_train.iloc[:, :-1]
    df_valid_features = df_valid.iloc[:, :-1]
    df_test_features = df_test.iloc[:, :-1]

    _mean = df_train_features.mean()
    _std = df_train_features.std()
    df_train_norm = (df_train_features - _mean) / _std
    df_valid_norm = (df_valid_features - _mean) / _std
    df_test_norm = (df_test_features - _mean) / _std

    df_train_norm.fillna(0, inplace=True)
    df_valid_norm.fillna(0, inplace=True)
    df_test_norm.fillna(0, inplace=True)


    df_train_norm['label'] = np.nan_to_num(labels[0],nan=0,posinf=0,neginf=0)
    df_valid_norm['label'] = np.nan_to_num(labels[1],nan=0,posinf=0,neginf=0)
    df_test_norm['label'] = np.nan_to_num(labels[2],nan=0,posinf=0,neginf=0)

    df_train_norm['label'] = df_train_norm['label'].groupby('datetime').transform(lambda x: (x - x.mean()) / x.std()).clip(-4, 4)

    # 所有数据 clip 到 [-4, 4]：减少异常值影响。
    df_train_norm = df_train_norm.clip(-4, 4)
    df_valid_norm = df_valid_norm.clip(-4, 4)
    df_test_norm = df_test_norm.clip(-4, 4)

    return df_train_norm, df_valid_norm, df_test_norm

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
    model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], 
                        num_boost_round=100, callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)])
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    pred = pd.concat([pd.Series(y_pred,index=df_test.index),df_test['label']],axis=1)

    return  model,pred

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
        'early_stopping_rounds': 10,
        'verbose_eval': 10
    }

    # Train the XGBoost model
    model = xgb.train(params, dtrain, evals=[(dtrain, 'train'), (dvalid, 'valid')], 
                      early_stopping_rounds=params['early_stopping_rounds'], verbose_eval=params['verbose_eval'])
    # Convert test data to DMatrix format
    dtest = xgb.DMatrix(X_test)

    # Make predictions on the test set
    y_pred = model.predict(dtest)
    # Combine the predictions with the actual labels
    pred = pd.concat([pd.Series(y_pred,index=df_test.index),df_test['label']],axis=1)

    return model, pred

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
    for _ in range(num_epochs):
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
        pred_df = pd.concat([pd.Series(test_outputs,index=df_test.index), df_test['label'],],axis=1)
    torch.cuda.empty_cache()
    return model, pred_df

def chunk_batch_spearmanr(x, y, chunk_size=100):
    n_days = len(x)
    spearmanr_list= []
    for i in range(0, n_days, chunk_size):
        spearmanr_list.append(batch_spearmanr(x[i:i+chunk_size], y[i:i+chunk_size]))
    spearmanr_list = torch.cat(spearmanr_list, dim=0)
    return spearmanr_list

def topk_dropn_returns(alpha: torch.Tensor, returns: torch.Tensor, k: int = 50, n: int = 5):
    """
    计算 TopK-DropN 策略的每日收益。
    
    Args:
        alpha: (T, N) 预测alpha值
        returns: (T, N) 股票真实日收益率
        k: 目标持仓数量
        n: 每日最大交易股票数（买卖合计）

    Returns:
        daily_returns: (T,) 每日策略收益（标量）
    """
    T, N = alpha.shape
    # 记录每日持仓（布尔 mask）
    current_holdings = torch.zeros(N, dtype=torch.bool)  # 初始空仓
    daily_pnl = []

    for t in range(T):
        # 获取当天 alpha 和收益，处理 NaN
        a_t = alpha[t].clone()
        r_t = returns[t].clone()
        valid_mask = ~torch.isnan(a_t) & ~torch.isnan(r_t)
        a_t[~valid_mask] = -float('inf')  # 无效股票排最后

        # 按 alpha 降序排序
        sorted_indices = torch.argsort(a_t, descending=True)
        topk_indices = sorted_indices[:k]

        # 构建目标持仓 mask
        target_holdings = torch.zeros(N, dtype=torch.bool)
        target_holdings[topk_indices] = True

        # 计算需要交易的股票（对称差集）
        to_buy = target_holdings & ~current_holdings
        to_sell = current_holdings & ~target_holdings
        candidates = torch.where(to_buy | to_sell)[0]

        # 如果候选交易数 <= n，全部执行
        if len(candidates) <= n:
            current_holdings = target_holdings.clone()
        else:
            # 超过 n，优先交易 alpha 差距最大的股票
            # 计算“交易优先级”：对要买/卖的股票，用 |alpha - 当前持仓阈值| 排序
            # 简化：直接按 alpha 排序选前 n 个（买高卖低）
            buy_candidates = to_buy.nonzero(as_tuple=True)[0]
            sell_candidates = to_sell.nonzero(as_tuple=True)[0]

            # 合并并按 alpha 重要性排序：买按 alpha 降序，卖按 alpha 升序（即卖最差的）
            # 更简单：把所有candidate按 alpha 绝对值排序（买高、卖低）
            candidate_alphas = a_t[candidates]
            # 买：alpha 越高越优先；卖：alpha 越低越优先 → 统一用 -alpha 排卖，+alpha 排买？
            # 替代方案：计算“偏离目标程度”
            trade_priority = torch.zeros_like(candidates, dtype=torch.float32)
            trade_priority[to_buy[candidates]] = a_t[candidates][to_buy[candidates]]  # 买高的优先
            trade_priority[to_sell[candidates]] = -a_t[candidates][to_sell[candidates]]  # 卖低的优先（-alpha 大）

            # 取优先级最高的 n 个
            _, top_n_idx = torch.topk(trade_priority, n)
            selected_candidates = candidates[top_n_idx]

            # 更新持仓：只交易 selected_candidates
            for idx in selected_candidates:
                current_holdings[idx] = target_holdings[idx]

        # 计算当日收益：等权持有 current_holdings 中的股票
        held_returns = r_t[current_holdings]
        held_returns = held_returns[~torch.isnan(held_returns)]
        if held_returns.numel() == 0:
            daily_ret = torch.tensor(0.0)
        else:
            daily_ret = held_returns.mean()  # 等权

        daily_pnl.append(daily_ret)

    return torch.stack(daily_pnl)

def get_tensor_metrics(x, y, risk_free_rate=0.0,transaction_cost:float=0.002,k:int=50,n:int=5):
    """
    x：表示因子的预测值，y：表示股票的收益率
    """
    # Ensure tensors are 2D (days, stocks)
    # 缩减维度
    if x.dim() > 2: x = x.squeeze(-1)
    if y.dim() > 2: y = y.squeeze(-1)

    # 计算每日IC值
    ic_s = batch_pearsonr(x, y)
    # 计算每日RankIC
    ric_s = chunk_batch_spearmanr(x, y, chunk_size=400)
    # 计算每一天因子值排名前20%的股票平均收益（策略收益，扣除成本）
    # gross_ret = batch_ret(x, y)
    gross_ret = topk_dropn_returns(x, y, k=k, n=n)  # 使用 TopK-DropN 交易策略
    net_ret = gross_ret - transaction_cost  # 扣除交易成本

    # 替换NaN值
    ic_s = torch.nan_to_num(ic_s, nan=0.)
    ric_s = torch.nan_to_num(ric_s, nan=0.)
    net_ret = torch.nan_to_num(net_ret, nan=0.)

    # 计算IC的均值
    ic_mean = ic_s.mean().item()
    # 计算IC的标准差
    ic_std = max(ic_s.std().item(), 1e-6)  # 避免除零
    # 计算Rank IC的均值
    ric_mean = ric_s.mean().item()
    # 计算Rank IC的标准差
    ric_std = max(ric_s.std().item(), 1e-6)

    # 计算ret的均值
    daily_ret_mean = net_ret.mean().item()
    # 计算ret的标准差
    daily_ret_std = max(net_ret.std().item(), 1e-6)

    # 计算衍生指标
    # ICIR=IC均值除以IC标准差
    icir = ic_mean / ic_std
    # RICIR=Rank IC均值除以Rank IC标准差
    ricir = ric_mean / ric_std
    retir = daily_ret_mean / daily_ret_std
 
    # 关键修正2：正确计算年化收益和夏普比率
    annualized_return = daily_ret_mean * 252  # 年化收益
    annualized_vol = daily_ret_std * (252 ** 0.5)  # 年化波动率

    # 计算夏普率
    ret_sharpe = batch_sharpe_ratio(net_ret, risk_free_rate).item()
    # 计算最大回撤
    ret_mdd = batch_max_drawdown(net_ret).item()
    result = dict(
        ic=ic_mean,
        ic_std=ic_std,
        icir=icir, 
        ric=ric_mean,
        ric_std=ric_std,
        ricir=ricir, 
        ret=annualized_return, # 年化收益
        ret_std=annualized_vol, # 年化波动率
        retir=retir,
        ret_sharpe=ret_sharpe,
        ret_mdd=ret_mdd,
    )
    return result, net_ret

def main(args):
    if args.instruments != 'sp500':
        train_start = 2011
        start_year = 2021
        end_year = 2024
    else:
        train_start = 2010
        start_year = 2016
        end_year = 2019

    for train_end in range(start_year,end_year):
        returned = get_data_by_year(
            train_start = train_start, train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
            instruments=args.instruments, target=target,freq='day',
            qlib_path=args.qlib_path
        )
        data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,_ = returned
        df_train = get_ml_data(data)
        df_valid = get_ml_data(data_valid)
        df_test = get_ml_data(data_test)
        df_train, df_valid, df_test = normalize_data(df_train, df_valid, df_test)
        
        model_name = 'lgbm'
        name = f"{args.instruments}_{model_name}_{train_end}"
        os.makedirs(f"out_ml/{name}",exist_ok=True)
        model,pred = train_lightgbm_model(df_train, df_valid, df_test)
        # 保存模型
        model.save_model(f"out_ml/{name}/{model_name}.pt")
        # 保存 pred与label
        pred.to_pickle(f"out_ml/{name}/pred.pkl")

    for train_end in range(start_year,end_year):
        returned = get_data_by_year(
            train_start = train_start,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
            instruments=args.instruments, target=target,freq='day',
            qlib_path=args.qlib_path
            )
        data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,_ = returned
        df_train = get_ml_data(data)
        df_valid = get_ml_data(data_valid)
        df_test = get_ml_data(data_test)
        df_train, df_valid, df_test = normalize_data(df_train, df_valid, df_test)
        
        model_name = 'xgb'
        name = f"{args.instruments}_{model_name}_{train_end}"
        os.makedirs(f"out_ml/{name}",exist_ok=True)
        model,pred = train_xgboost_model(df_train, df_valid, df_test)
        model.save_model(f"out_ml/{name}/{model_name}.pt")
        pred.to_pickle(f"out_ml/{name}/pred.pkl")


    # # Train MLP Model
    for train_end in range(start_year,end_year):
        returned = get_data_by_year(
            train_start = train_start,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
            instruments=args.instruments, target=target,freq='day',)
        data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,_ = returned
        df_train = get_ml_data(data)
        df_valid = get_ml_data(data_valid)
        df_test = get_ml_data(data_test)
        df_train, df_valid, df_test = normalize_data(df_train, df_valid, df_test)
        for seed in range(1):
            reseed_everything(seed)
            model_name = 'mlp'
            name = f"{args.instruments}_{model_name}_{train_end}"
            os.makedirs(f"out_ml/{name}",exist_ok=True)
            model,pred = train_mlp_model(df_train, df_valid, df_test)
            # model.save_model(f"out_ml/{name}/model.pt")
            pred.to_pickle(f"out_ml/{name}/pred_{seed}.pkl")


    # # Show LightGBM Resuls
    result = []
    for year in range(start_year,end_year):
        result.append(pd.read_pickle(f'out_ml/{args.instruments}_lgbm_{year}/pred.pkl'))

    df = pd.concat(result,axis=0)
    data = df.pivot_table(index="datetime", columns="instrument", values=[0,"label"])
    pred = data[0].values
    label = data["label"].values
    res, ret_s = get_tensor_metrics(torch.tensor(pred), torch.tensor(label))
    print('=' * 66)
    print("Show LightGBM Resuls")
    print(pd.DataFrame(res,index=['Test']))
    # 保存到汇总目录（不依赖 year）
    summary_dir = f"out_ml/{args.instruments}_lgbm_summary"
    os.makedirs(summary_dir, exist_ok=True)
    np.save(os.path.join(summary_dir, 'ret_s.npy'), ret_s)
    pd.DataFrame(res, index=['Test']).to_csv(os.path.join(summary_dir, 'metrics.csv'))


    # # Show XGBoost Result
    result = []
    for year in range(start_year,end_year):
        result.append(pd.read_pickle(f'out_ml/{args.instruments}_xgb_{year}/pred.pkl'))
    df = pd.concat(result,axis=0)

    data = df.pivot_table(index="datetime", columns="instrument", values=[0,"label"])
    pred = data[0].values
    label = data["label"].values
    res, ret_s = get_tensor_metrics(torch.tensor(pred), torch.tensor(label))
    print('=' * 66)
    print("Show XGBoost Result")
    print(pd.DataFrame(res,index=['Test']))
    # 保存到汇总目录（不依赖 year）
    summary_dir = f"out_ml/{args.instruments}_xgb_summary"
    os.makedirs(summary_dir, exist_ok=True)
    np.save(os.path.join(summary_dir, 'ret_s.npy'), ret_s)
    pd.DataFrame(res, index=['Test']).to_csv(os.path.join(summary_dir, 'metrics.csv'))


    # # Show MLP Result
    for seed in range(1):
        result = []
        for year in range(start_year,end_year):
            result.append(pd.read_pickle(f'out_ml/{args.instruments}_mlp_{year}/pred_{seed}.pkl'))
        df = pd.concat(result,axis=0)
        data = df.pivot_table(index="datetime", columns="instrument", values=[0,"label"])
        pred = data[0].values
        label = data["label"].values
        res, ret_s = get_tensor_metrics(torch.tensor(pred), torch.tensor(label))
        print('=' * 66)
        print("Show MLP Result")
        print(pd.DataFrame(res,index=['Test']))
        # 保存到汇总目录（不依赖 year）
        summary_dir = f"out_ml/{args.instruments}_mlp_summary"
        os.makedirs(summary_dir, exist_ok=True)
        np.save(os.path.join(summary_dir, 'ret_s.npy'), ret_s)
        pd.DataFrame(res, index=['Test']).to_csv(os.path.join(summary_dir, 'metrics.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Alpha Pool Evaluation with Configurable Parameters")
    parser.add_argument("--instruments", type=str, default="sp500", help="Instrument universe (e.g., 'sp500')")
    parser.add_argument("--qlib_path", type=str, required=True, help="Path to Qlib data directory")
    args = parser.parse_args()
    main(args)
