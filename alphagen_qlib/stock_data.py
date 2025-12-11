from typing import List, Union, Optional, Tuple, Dict
from enum import IntEnum
import numpy as np
import pandas as pd
import torch

class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5

def change_to_raw_min(features):
    result = []
    for feature in features:
        if feature in ['$vwap']:
            result.append(f"$money/$volume")
        elif feature in ['$volume']:
            result.append(f"{feature}/100000")
            # result.append('$close')
        else:
            result.append(feature)
    return result

def change_to_raw(features):
    result = []
    for feature in features:
        if feature in ['$open','$close','$high','$low','$vwap']:
            result.append(f"{feature}*$factor")
        elif feature in ['$volume']:
            result.append(f"{feature}/$factor/1000000")
            # result.append('$close')
        else:
            raise ValueError(f"feature {feature} not supported")
    return result

class StockData:
    _qlib_initialized: bool = False

    def __init__(self,
                 instrument: Union[str, List[str]],
                 start_time: str,
                 end_time: str,
                 max_backtrack_days: int = 100,
                 max_future_days: int = 30,
                 features: Optional[List[FeatureType]] = None,
                 device: torch.device = torch.device('cuda:0'),
                 raw:bool = False,
                 qlib_path:Union[str,Dict] = "",
                 freq:str = 'day',
                 ) -> None:
        self._init_qlib(qlib_path)
        self.df_bak = None
        self.raw = raw
        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.freq = freq
        self.data, self._dates, self._stock_ids = self._get_data()

    @classmethod
    def _init_qlib(cls,qlib_path) -> None:
        if cls._qlib_initialized:
            return
        import qlib
        from qlib.config import REG_CN, REG_US
        if 'us' in qlib_path:
            region = REG_US
        else:
            region = REG_CN
        print(f"qlib_path: {qlib_path}\t region: {region}")
        qlib.init(provider_uri=qlib_path, region=region)
        cls._qlib_initialized = True

    # def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
    #     # This evaluates an expression on the data and returns the dataframe
    #     # It might throw on illegal expressions like "Ref(constant, dtime)"
    #     from qlib.data.dataset.loader import QlibDataLoader
    #     from qlib.data import D
    #     if not isinstance(exprs, list):
    #         exprs = [exprs]
    #     cal: np.ndarray = D.calendar()
    #     start_index = cal.searchsorted(pd.Timestamp(self._start_time))  # type: ignore
    #     end_index = cal.searchsorted(pd.Timestamp(self._end_time))  # type: ignore
    #     real_start_time = cal[start_index - self.max_backtrack_days]
    #     if cal[end_index] != pd.Timestamp(self._end_time):
    #         end_index -= 1
    #     real_end_time = cal[end_index + self.max_future_days]
    #     print(f'real_start_time: {real_start_time}')
    #     print(f'real_end_time: {real_end_time}')
    #     return
    #     return (QlibDataLoader(config=exprs).load(self._instrument, real_start_time, real_end_time))
    
    def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
        # This evaluates an expression on the data and returns the dataframe
        # It might throw on illegal expressions like "Ref(constant, dtime)"
        from qlib.data.dataset.loader import QlibDataLoader
        from qlib.data import D
        if not isinstance(exprs, list):
            exprs = [exprs]
        cal: np.ndarray = D.calendar(freq=self.freq)
        # 获取用户请求的时间（转为pandas Timestamp）
        req_start = pd.Timestamp(self._start_time)
        req_end = pd.Timestamp(self._end_time)

        # 安全裁剪：不能早于日历最早日，也不能晚于日历最晚日
        cal_start = pd.Timestamp(cal[0])
        cal_end = pd.Timestamp(cal[-1])
        if req_start > cal_end:
            raise ValueError(f"Requested start_time {req_start} is after the latest available data {cal_end}.")
        if req_end < cal_start:
            raise ValueError(f"Requested end_time {req_end} is before the earliest available data {cal_start}.")

        # 裁剪到有效范围
        effective_start = max(req_start, cal_start)
        effective_end = min(req_end, cal_end)

        # 找到对应索引
        start_index = cal.searchsorted(effective_start, side="left")
        end_index = cal.searchsorted(effective_end, side="right") - 1
        #  扩展时间窗口（用于特征计算）
        real_start_index = max(0, start_index - self.max_backtrack_days)
        real_end_index = min(len(cal) - 1, end_index + self.max_future_days)

        real_start_time = cal[real_start_index]
        real_end_time = cal[real_end_index]
        print(f'real_start_time: {real_start_time}')
        print(f'real_end_time: {real_end_time}')
        # 加载数据
        result = (QlibDataLoader(config=exprs, freq=self.freq)
                .load(self._instrument, real_start_time, real_end_time))
        
        if result.empty:
            raise ValueError(f"No data loaded for instrument(s) {self._instrument} "
                            f"between {real_start_time} and {real_end_time}.")

        return result

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        features = ['$' + f.name.lower() for f in self._features]
        if self.raw and self.freq == 'day':
            features = change_to_raw(features)
        elif self.raw:
            features = change_to_raw_min(features)
        df = self._load_exprs(features)
        self.df_bak = df
        # print(df)
        # print("Original df shape:", df.shape)
        # print("Original df index levels:", df.index.nlevels)
        # print("Original df columns:", df.columns[:5])  # 查看列结构
        # 注释这4行
        # df = df.stack().unstack(level=1)
        # dates = df.index.levels[0] 
        # stock_ids = df.columns
        # values = df.values 
        # print("After stack().unstack(level=1):")
        # print("df shape:", df.shape)
        # print("df index:", df.index)
        # print("df columns[:5]:", df.columns[:5])
        # 修改下面4行                                    # type: ignore
        df_unstacked = df.unstack(level=1)
        dates = df_unstacked.index
        stock_ids = df.index.get_level_values(1).unique()
        values = df_unstacked.values.reshape(len(dates), len(features), len(stock_ids))
        values = values.reshape((-1, len(features), values.shape[-1]))  # type: ignore
        return torch.tensor(values, dtype=torch.float, device=self.device), dates, stock_ids

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def add_data(self,data:torch.Tensor,dates:pd.Index):
        data = data.to(self.device)
        self.data = torch.cat([self.data,data],dim=0)
        self._dates = pd.Index(self._dates.append(dates))


    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
            Parameters:
            - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
            a list of tensors of size `(n_days, n_stocks)`
            - `columns`: an optional list of column names
            """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
