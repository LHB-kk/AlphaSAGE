
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import json
import torch
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen_generic.features import *
from gan.utils.data import get_data_by_year

from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr, batch_ret, batch_sharpe_ratio, batch_max_drawdown

def pred_pool(capacity,data,cache):
    from alphagen_qlib.calculator import QLibStockDataCalculator
    pool = AlphaPool(capacity=capacity,
                    stock_data=data,
                    target=target,
                    ic_lower_bound=None)
    exprs = []
    for key in dict(Counter(cache).most_common(capacity)):
        exprs.append(eval(key))
    pool.force_load_exprs(exprs)
    pool._optimize(alpha=5e-3, lr=5e-1, n_iter=2000)

    exprs = pool.exprs[:pool.size]
    weights = pool.weights[:pool.size]
    calculator_test = QLibStockDataCalculator(data, target)
    ensemble_value = calculator_test.make_ensemble_alpha(exprs, weights)
    return ensemble_value

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
    ret_s = batch_ret(x, y) -0.003

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

def main(args):
    if args.instruments != 'sp500':
        train_start = 2011
        strat_year = 2021
        end_year = 2024
    else:
        train_start = 2010
        strat_year = 2016
        end_year = 2019

    for seed in range(1):
        for train_end in range(strat_year,end_year):
            for num in [1]:
                save_dir = f'out_gp/{args.instruments}_{train_end}_day_{seed}' 
                print(save_dir)
                
                returned = get_data_by_year(
                    train_start = train_start,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
                    instruments=args.instruments, target=target,freq='day',
                    qlib_path = args.qlib_path
                )
                data_all,data,data_valid,data_valid_withhead,data_test,data_test_withhead,name = returned
                cache = json.load(open(f'{save_dir}/2.json'))['cache']

                features = ['open_', 'close', 'high', 'low', 'volume', 'vwap']
                constants = [f'Constant({v})' for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]]
                terminals = features + constants

                pred = pred_pool(num,data_all,cache)
                pred = pred[-data_test.n_days:]
                torch.save(pred.detach().cpu(),f"{save_dir}/pred_{num}.pt")


    all_preds = []   # 收集所有预测
    all_tgts = []    # 收集所有真实目标

    for seed in range(1):
        for train_end in range(strat_year, end_year):
            for num in [1]:
                save_dir = f'out_gp/{args.instruments}_{train_end}_day_{seed}'
                returned = get_data_by_year(
                    train_start=train_start,
                    train_end=train_end,
                    valid_year=train_end + 1,
                    test_year=train_end + 2,
                    instruments=args.instruments,
                    target=target,
                    freq='day',
                    qlib_path=args.qlib_path
                )
                data_all, data, data_valid, data_valid_withhead, data_test, data_test_withhead, name = returned

                # 加载预测结果
                pred = torch.load(f"{save_dir}/pred_{num}.pt")  # shape: (n_days,)
                pred = torch.tensor(pred).float()  # 确保是 tensor

                # 获取对应的真实目标
                tgt = target.evaluate(data_all)[-data_test.n_days:, :]  # shape: (n_days, 1) 或 (n_days,)
                tgt = tgt.to("cpu").float().squeeze()  # 转为 1D，确保和 pred 一致

                # 安全检查
                assert pred.shape == tgt.shape, f"Shape mismatch: pred={pred.shape}, tgt={tgt.shape} in {save_dir}"

                # 收集
                all_preds.append(pred)
                all_tgts.append(tgt)

    # ===== 循环结束后，统一计算 =====
    if all_preds:
        # 拼接所有预测和目标
        total_pred = torch.cat(all_preds, dim=0)  # shape: (total_days,)
        total_tgt = torch.cat(all_tgts, dim=0)    # shape: (total_days,)

        # 计算整体指标
        res, ret_s = get_tensor_metrics(total_pred, total_tgt)
        df_res = pd.DataFrame([res], index=["Overall"])
        print(df_res)

        # 保存整体结果
        np.save("out_gp/overall_ret_s.npy", ret_s)
    else:
        print("No data collected!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Alpha Pool Evaluation with Configurable Parameters")
    parser.add_argument("--instruments", type=str, default="sp500", help="Instrument universe (e.g., 'sp500')")
    parser.add_argument("--qlib_path", type=str, required=True, help="Path to Qlib data directory")
    args = parser.parse_args()
    main(args)





