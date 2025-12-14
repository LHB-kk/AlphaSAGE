
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import json
import torch
import argparse
import pandas as pd

# from alphagen.config import *
# from alphagen.data.tokens import *
from alphagen_generic.features import open_
from alphagen_generic.features import *
from alphagen.data.expression import *
from alphagen.utils.correlation import batch_pearsonr,batch_spearmanr
from typing import Tuple
from gan.utils.data import get_data_by_year
from alphagen_qlib.calculator import QLibStockDataCalculator

def load_alpha_pool(raw) -> Tuple[List[Expression], List[float]]:
    exprs_raw = raw['exprs']
    exprs = [eval(expr_raw.replace('open', 'open_').replace('$', '')) for expr_raw in exprs_raw]
    weights = raw['weights']
    return exprs, weights

def load_alpha_pool_by_path(path: str) -> Tuple[List[Expression], List[float]]:
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
        return load_alpha_pool(raw)
    
def load_ppo_path(path,name_prefix):
    
    files = os.listdir(path)
    folder = [i for i in files if name_prefix in i][-1]
    names = [i for i in os.listdir(f"{path}/{folder}") if '.json' in i]
    name = sorted(names,key = lambda x:int(x.split('_')[0]))[-1]
    return f"{path}/{folder}/{name}"

def main(args):
    # # infer
    freq = 'day'
    chk_path = "out_ppo/checkpoints"
    device = 'cuda:0'

    if args.instruments != 'sp500':
        train_start = 2011
        start_year = 2021
        end_year = 2024
    else:
        train_start = 2010
        start_year = 2016
        end_year = 2019

    result = []
    for train_end in range(start_year,end_year):
        returned = get_data_by_year(
            train_start = train_start,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
            instruments=args.instruments, target=target,freq=freq,
        )
        data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,name = returned
        for seed in range(5):
            for num in [1,10,20,50,100]:
                name_prefix = f"{args.instruments}_{train_end}_{num}_{seed}"
                path = load_ppo_path(chk_path,name_prefix)
                    
                exprs,weights = load_alpha_pool_by_path(path)
                
                # calculator_test = QLibStockDataCalculator(data_test, target)
                calculator_test = QLibStockDataCalculator(data_all, target)

                ensemble_value = calculator_test.make_ensemble_alpha(exprs, weights)
                ensemble_value = ensemble_value[-data_test.n_days:]
                dirname = os.path.dirname(path)
                
                torch.save(ensemble_value.cpu(),f"{dirname}/{train_end}_{num}_{seed}.pkl")


    # # read the infer result and evaluate
    result = []
    for seed in range(5):
        cur_seed_ic = []
        cur_seed_ric = []
        for num in [50,100]:
            for train_end in range(start_year,end_year):
                returned = get_data_by_year(
                    train_start = train_start,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
                    instruments=args.instruments, target=target,freq=freq,
                )
                data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,name = returned

                
                name_prefix = f"n1230day_{args.instruments}_{train_end}_{num}_{seed}"
                path = load_ppo_path(chk_path,name_prefix)
                dirname = os.path.dirname(path)
                pred = torch.load(f"{dirname}/{train_end}_{num}_{seed}.pkl").to(device)
                tgt = target.evaluate(data_test)
                tgt = target.evaluate(data_all)[-data_test.n_days:,:]

                ic_s = torch.nan_to_num(batch_pearsonr(pred,tgt),nan=0)
                rank_ic_s = torch.nan_to_num(batch_spearmanr(pred,tgt),nan=0)

                cur_seed_ic.append(ic_s)
                cur_seed_ric.append(rank_ic_s)
            ic = torch.cat(cur_seed_ic)
            rank_ic = torch.cat(cur_seed_ric)

            ic_mean = ic.mean().item()
            rank_ic_mean = rank_ic.mean().item()
            ic_std = ic.std().item()
            rank_ic_std = rank_ic.std().item()
            tmp = dict(
                seed = seed,
                num = num,
                ic = ic_mean,
                ric = rank_ic_mean,
                icir = ic_mean/ic_std,
                ricir = rank_ic_mean/rank_ic_std,
            )
            result.append(tmp)

    exp_result = pd.DataFrame(result).groupby(['num','seed']).mean().groupby('num').agg(['mean','std'])
    print(exp_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Alpha Pool Evaluation with Configurable Parameters")
    parser.add_argument("--instruments", type=str, default="sp500", help="Instrument universe (e.g., 'sp500')")
    parser.add_argument("--qlib_path", type=str, required=True, help="Path to Qlib data directory")
    args = parser.parse_args()
    main(args)
            


