import os
import yaml
import argparse
import torch
from datetime import datetime

from fqf_iqn_qrdqn.agent import QRQCMAgent, IQCMAgent, FQCMAgent
from alphagen.data.expression import Feature, FeatureType, Ref, StockData
from alphagen.models.alpha_pool import AlphaPool
from alphagen.rl.env.wrapper import AlphaEnv

def run(args):
    config_path = os.path.join('qcm_config', f'{args.model}.yaml')

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    device = torch.device(f'cuda')
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    
    if args.instruments != "sp500":
        data_train = StockData(instrument=args.instruments,
                    start_time='2011-01-01',
                    end_time='2021-12-31',
                    qlib_path = args.qlib_path)
        data_valid = StockData(instrument=args.instruments,
                    start_time='2022-01-01',
                    end_time='2022-12-31',
                    qlib_path = args.qlib_path)
        data_test = StockData(instrument=args.instruments,
                    start_time='2023-01-01',
                    end_time='2025-12-31',
                    qlib_path = args.qlib_path)
    else:
        data_train = StockData(instrument=args.instruments,
                    start_time='2010-01-01',
                    end_time='2016-12-31',
                    qlib_path = args.qlib_path)
        data_valid = StockData(instrument=args.instruments,
                        start_time='2017-01-01',
                        end_time='2017-12-31',
                        qlib_path = args.qlib_path)
        data_test = StockData(instrument=args.instruments,
                        start_time='2018-01-01',
                        end_time='2020-12-31',
                        qlib_path = args.qlib_path)

    train_pool = AlphaPool(
        capacity=args.pool,
        stock_data=data_train,
        target=target,
        ic_lower_bound=None
    )
    train_env = AlphaEnv(pool=train_pool, device=device, print_expr=True)

    # Specify the directory to log.
    name = args.model
    time = datetime.now().strftime("%Y%m%d-%H%M")
    if name in ['qrdqn', 'iqn']:
        log_dir = os.path.join(f'data/{args.instruments}_logs',
                           f'pool_{args.pool}_QCM_{args.std_lam}',
                           f"{name}-seed{args.seed}-{time}-N{config['N']}-lr{config['lr']}-per{config['use_per']}-gamma{config['gamma']}-step{config['multi_step']}")
    elif name == 'fqf':
        log_dir = os.path.join(f'data/{args.instruments}_logs',
                           f'pool_{args.pool}_QCM_{args.std_lam}',
                           f"{name}-seed{args.seed}-{time}-N{config['N']}-lr{config['quantile_lr']}-per{config['use_per']}-gamma{config['gamma']}-step{config['multi_step']}")

    # Create the agent and run.
    if name == 'qrdqn':
        agent = QRQCMAgent(env=train_env,
                           data_valid=data_valid,
                           data_test=data_test,
                           target=target,
                           log_dir=log_dir,
                           seed=args.seed,
                           std_lam=args.std_lam,
                           cuda=True,
                           **config)
    elif name == 'iqn':
        agent = IQCMAgent(env=train_env,
                          data_valid=data_valid,
                          data_test=data_test,
                          target=target,
                          log_dir=log_dir,
                          seed=args.seed,
                          std_lam=args.std_lam,
                          cuda=True, **config)
    elif name == 'fqf':
        agent = FQCMAgent(env=train_env,
                          data_valid=data_valid,
                          data_test=data_test,
                          target=target,
                          log_dir=log_dir,
                          seed=args.seed,
                          std_lam=args.std_lam,
                          cuda=True, **config)
        
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qrdqn')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pool', type=int, default=20)
    parser.add_argument('--std-lam', type=float, default=1.0)
    parser.add_argument('--instruments', type=str, default='csi300')
    parser.add_argument('--qlib_path', type=str, default='')
    args = parser.parse_args()
    run(args)
