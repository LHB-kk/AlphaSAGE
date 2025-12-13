
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import json
import torch 
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

instruments: str = "csi300"

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

# # infer
freq = 'day'
chk_path = "out_ppo/checkpoints"
result = []
for train_end in range(2016,2021):
    returned = get_data_by_year(
        train_start = 2010,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
        instruments=instruments, target=target,freq='day',
    )
    data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,name = returned
    for seed in range(5):
        for num in [1,10,20,50,100]:
            name_prefix = f"csi300_{train_end}_{num}_{seed}"
            path = load_ppo_path(chk_path,name_prefix)
                
            exprs,weights = load_alpha_pool_by_path(path)
            
            # calculator_test = QLibStockDataCalculator(data_test, target)
            calculator_test = QLibStockDataCalculator(data_all, target)

            ensemble_value = calculator_test.make_ensemble_alpha(exprs, weights)
            ensemble_value = ensemble_value[-data_test.n_days:]
            dirname = os.path.dirname(path)
            
            torch.save(ensemble_value.cpu(),f"{dirname}/{train_end}_{num}_{seed}.pkl")


# # read the infer result and evaluate
device = 'cuda:0'
result = []
for seed in range(5):
    cur_seed_ic = []
    cur_seed_ric = []
    
    for num in [50,100]:
        for train_end in range(2016,2021):
            returned = get_data_by_year(
                train_start = 2010,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
                instruments=instruments, target=target,freq=freq,
            )
            data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,name = returned

            
            name_prefix = f"n1230day_csi500_{train_end}_{num}_{seed}"
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
            


