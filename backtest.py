from typing import Optional, TypeVar, Callable, Optional, Tuple
import os
import pickle
import warnings
import pandas as pd
import json
from pathlib import Path
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin

from qlib.backtest import backtest, executor as exec
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report.analysis_position import report_graph
from qlib.contrib.strategy import TopkDropoutStrategy

from alphagen.data.expression import *
from alphagen_generic.features import *
from alphagen_qlib.stock_data import StockData
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.utils import load_alpha_pool_by_path


_T = TypeVar("_T")


def _create_parents(path: str) -> None:
    """判断父目录是否存在，不存在则创建父目录"""
    dir = os.path.dirname(path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)


def write_all_text(path: str, text: str) -> None:
    """写入文本文件，如果父目录不存在则创建父目录"""
    _create_parents(path)
    with open(path, "w") as f:
        f.write(text)


def dump_pickle(path: str,
                factory: Callable[[], _T],
                invalidate_cache: bool = False) -> Optional[_T]:
    if invalidate_cache or not os.path.exists(path):
        _create_parents(path)
        obj = factory()
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return obj


@dataclass
class BacktestResult(DataClassJsonMixin):
    sharpe: float
    annual_return: float
    max_drawdown: float
    information_ratio: float
    annual_excess_return: float
    excess_max_drawdown: float
    cumulative_return: float
    excess_cumulative_return: float


class QlibBacktest:
    def __init__(
        self,
        benchmark: str = "SH000300",
        top_k: int = 50,
        n_drop: Optional[int] = None,
        deal: str = "close",
        open_cost: float = 0.0015,
        close_cost: float = 0.0015,
        min_cost: float = 5,
    ):
        self._benchmark = benchmark
        self._top_k = top_k
        self._n_drop = n_drop if n_drop is not None else top_k
        self._deal_price = deal
        self._open_cost = open_cost
        self._close_cost = close_cost
        self._min_cost = min_cost

    def run(
        self,
        prediction: Union[pd.Series, pd.DataFrame],
        output_prefix: Optional[str] = None
    ) -> Tuple[pd.DataFrame, BacktestResult]:
        prediction = prediction.sort_index()
        index: pd.MultiIndex = prediction.index.remove_unused_levels()  # type: ignore
        dates = index.levels[0]

        def backtest_impl(last: int = -1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                strategy = TopkDropoutStrategy(
                    signal=prediction,
                    topk=self._top_k,
                    n_drop=self._n_drop,
                    only_tradable=True,
                    forbid_all_trade_at_limit=True
                )
                executor = exec.SimulatorExecutor(
                    time_per_step="day",
                    generate_portfolio_metrics=True
                )
                return backtest(
                    strategy=strategy,
                    executor=executor,
                    start_time=dates[0],
                    end_time=dates[last],
                    account=100_000_000,
                    benchmark=self._benchmark,
                    exchange_kwargs={
                        "limit_threshold": 0.095,
                        "deal_price": self._deal_price,
                        "open_cost": self._open_cost,
                        "close_cost": self._close_cost,
                        "min_cost": self._min_cost,
                    }
                )[0]

        try:
            portfolio_metric = backtest_impl()
        except IndexError:
            print("Cannot backtest till the last day, trying again with one less day")
            portfolio_metric = backtest_impl(-2)

        report, _ = portfolio_metric["1day"]    # type: ignore
        result = self._analyze_report(report)
        graph = report_graph(report, show_notebook=False)[0]
        if output_prefix is not None:
            dump_pickle(output_prefix + "-report.pkl", lambda: report, True)
            dump_pickle(output_prefix + "-graph.pkl", lambda: graph, True)
            write_all_text(output_prefix + "-result.json", result.to_json())
        return report, result

    def _analyze_report(self, report: pd.DataFrame) -> BacktestResult:
        net_return = report["return"] - report["cost"]
        returns_analysis = risk_analysis(net_return)["risk"]
        cum_return = returns_analysis.loc["cumulative_returns"]

        excess_return = net_return - report["bench"]
        excess_analysis = risk_analysis(excess_return)["risk"]
        cum_excess_return = excess_analysis.loc["cumulative_returns"]

        # 绘制累计收益曲线
        # import matplotlib.pyplot as plt

        # net_return = report["return"] - report["cost"]
        # cumulative_nav = (1 + net_return).cumprod()

        # plt.figure(figsize=(10, 6))
        # plt.plot(cumulative_nav.index, cumulative_nav.values)
        # plt.title("Cumulative Net Asset Value (NAV)")
        # plt.ylabel("NAV")
        # plt.xlabel("Date")
        # plt.grid(True)
        # plt.show()

        return BacktestResult(
            sharpe=returns_analysis.loc["sharpe"],  # 修正：使用 sharpe 而非 information_ratio
            annual_return=returns_analysis.loc["annualized_return"],
            max_drawdown=returns_analysis.loc["max_drawdown"],
            information_ratio=excess_analysis.loc["information_ratio"],
            annual_excess_return=excess_analysis.loc["annualized_return"],
            excess_max_drawdown=excess_analysis.loc["max_drawdown"],
            # 如果 BacktestResult 添加了新字段：
            cumulative_return=cum_return,
            excess_cumulative_return=cum_excess_return,
        )


if __name__ == "__main__":
    qlib_backtest = QlibBacktest(top_k=50, n_drop=5)
    data = StockData(
        instrument="csi300",
        start_time="2022-01-01",
        end_time="2023-06-30",
        qlib_path=""
    )
    calc = QLibStockDataCalculator(data, None)

    def run_backtest(prefix: str, seed: int, exprs: List[Expression], weights: List[float]):
        df = data.make_dataframe(calc.make_ensemble_alpha(exprs, weights))
        qlib_backtest.run(df, output_prefix=f"out/backtests/50-5/{prefix}/{seed}")

    for p in Path("out/results").iterdir():
        # 如果p是文件，并且后缀名为.json
        if p.is_file() and p.suffix == ".json":
            ver, inst, size, seed, time = p.name.split('_', 4)
            size, seed = int(size), int(seed)
            exprs, weights = load_alpha_pool_by_path(str(p / "251904_steps_pool.json"))
            run_backtest(ver, seed, exprs, weights)
