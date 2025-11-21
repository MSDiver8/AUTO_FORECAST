import heapq as hq
import warnings
from itertools import product
from math import inf, sqrt
from typing import Literal

import arch.unitroot as uroot
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from arch.utility.exceptions import InfeasibleTestException
from numpy.linalg import LinAlgError
from statsmodels.genmod.cov_struct import ConvergenceWarning
from statsmodels.api import OLS
from statsmodels.tsa.api import arima
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import InterpolationWarning, acf

# import hegy

warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=InterpolationWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

SEASONS = {
    "Y": 1,
    "YE": 1,
    "Q": 4,
    "QE": 4,
    "M": 12,
    "ME": 12,
    "W": 52,
    "WE": 52,
    "D": 365,
}
LOWLIMITS = {
    "Y": 15,
    "YE": 15,
    "Q": 16,
    "QE": 16,
    "M": 24,
    "ME": 24,
    "W": 24,
    "WE": 24,
    "D": 24,
}


class GetARIMA:
    def __init__(
        self,
        data: dict,
        exog: np.ndarray | pd.Series | pd.DataFrame | None = None,
        max_order: int = 5,
        max_d: int = 2,
        max_sorder: int = 1,
        max_D: int = 1,
        seasonal_strength: float = 0.64,
        box_cox: float | Literal["auto"] | None = None,
        criteria: str = "aic",
        lag_criteria: str | None = None,
    ) -> None:
        """Инициализация объекта GetARIMA.

        Keyword arguments:
        data -- словарь-обертка вокруг данных.
        exog -- NumPy-массив или Pandas-Series с экзогенными переменными.
        max_order -- максимальное значение порядка авторегрессии и скользящего среднего.
        max_sorder -- максимальное значение порядка сезонных авторегрессии и скользящего среднего.
        max_d -- максимальный порядок интегрированности.
        max_D -- максимальный порядок сезонной интегрированности.
        seasonal_strength -- пороговый показатель силы сезонности.
        box_cox -- показатель lambda для преобразования Бокса-Кокса.
        criteria -- информационный критерий для выбора модели.
        lag_criteria -- информационный критерий для выбора числа лагов.
                        Если не задано, то принимается равным criteria
        """
        if lag_criteria is None:
            lag_criteria = criteria

        assert criteria in ("aic", "bic", "hqic"), "Неверный IC!"
        assert lag_criteria in ("aic", "bic", "hqic", "fixed"), (
            "Неверный IC для выбора лага!"
        )
        assert max_D in (0, 1), "D больше, чем 1, не поддерживается!"

        self.nobs = data["observations_count"]

        self.y = pd.DataFrame(data["observations"])
        self.y.index = pd.to_datetime(self.y.date, format="%d.%m.%Y")
        self.y.obs = self.y.obs.apply(float)
        self.y.drop(["date"], axis=1, inplace=True)

        if box_cox is not None:
            if box_cox == "auto":
                self.y.loc[:, "obs"] = boxcox(self.y.obs)[0]
            else:
                self.y.loc[:, "obs"] = boxcox(self.y.obs, lmbda=box_cox)[0]

        if exog is not None:
            if isinstance(exog, np.ndarray):
                if len(exog.shape) == 1:
                    X = exog[:, np.newaxis]
            elif isinstance(exog, pd.Series):
                X = np.asarray(exog.values)[:, np.newaxis]
            elif isinstance(exog, pd.DataFrame):
                X = np.asarray(exog.values)

            n, _ = X.shape
            if n != self.nobs:
                raise ValueError(
                    "Число наблюдений у эндогенной и экзогенных переменных не совпадает!"
                )
            self.y.obs = OLS(self.y.obs.values, X).fit().residuals

        self.criteria = criteria
        self.lag_criteria = lag_criteria if lag_criteria is not None else criteria

        self.freq = data["pandas_frequency"]

        if self.freq is None:
            self.freq = "Y"

        assert self.nobs >= max(2 * SEASONS[self.freq], LOWLIMITS[self.freq]), (
            "Слишком короткий временной ряд!"
        )

        self.max_order = max_order
        self.max_d = max_d

        self.max_sorder = max_sorder
        self.max_D = max_D

    def _next_step(self, model: tuple) -> list:
        (p, d, q), (P, D, Q, s), trend = model

        next_list = [((p, d, q), (P, D, Q, s), not trend)]

        if p < self.max_order:
            next_list.append(((p + 1, d, q), (P, D, Q, s), trend))
        if q < self.max_order:
            next_list.append(((p, d, q + 1), (P, D, Q, s), trend))
        if p > 0:
            next_list.append(((p - 1, d, q), (P, D, Q, s), trend))
        if q > 0:
            next_list.append(((p, d, q - 1), (P, D, Q, s), trend))

        if s:
            if P < self.max_sorder:
                next_list.append(((p, d, q), (P + 1, D, Q, s), trend))
            if Q < self.max_sorder:
                next_list.append(((p, d, q), (P, D, Q + 1, s), trend))
            if P > 0:
                next_list.append(((p, d, q), (P - 1, D, Q, s), trend))
            if Q > 0:
                next_list.append(((p, d, q), (P, D, Q - 1, s), trend))

        if p < self.max_order and q < self.max_order:
            next_list.append(((p + 1, d, q + 1), (P, D, Q, s), trend))
        if p > 0 and q > 0:
            next_list.append(((p - 1, d, q - 1), (P, D, Q, s), trend))

        if s:
            if P < self.max_sorder and Q < self.max_sorder:
                next_list.append(((p, d, q), (P + 1, D, Q + 1, s), trend))
            if P > 0 and Q > 0:
                next_list.append(((p, d, q), (P - 1, D, Q - 1, s), trend))

        return next_list

    def _get_dd(self) -> tuple[int, int]:
        y = self.y.obs.copy()

        if (self.freq not in ("M", "ME", "Q", "QE")) or (self.max_D == 0):
            D = 0
        else:
            seasons = 4 if self.freq == "Q" else 12

            D = int(self._seasonal_strength(y) > 0.64)

            if D > 0:
                y = (y - y.shift(seasons)).dropna()

        d = 0
        while d <= self.max_d:
            votes = 0
            voters = 5

            try:
                _stat = 1
                for trend in ("n", "c", "ct"):
                    _adf = uroot.ADF(
                        y,
                        trend=trend,
                        method=self.lag_criteria,  # pyright: ignore[reportArgumentType]
                    )
                    _stat = min(_stat, _adf.pvalue)
                if _stat > 0.05:
                    votes += 1
            except InfeasibleTestException:
                voters -= 1

            try:
                _stat = 1
                for trend in ("c", "ct"):
                    _kpss = uroot.KPSS(
                        y,
                        trend=trend,
                    )
                    _stat = min(_stat, _kpss.pvalue)
                if _stat < 0.05:
                    votes += 1
            except InfeasibleTestException:
                voters -= 1

            try:
                _stat = 1
                for trend in ("n", "c", "ct"):
                    _pp = uroot.PhillipsPerron(
                        y,
                        trend=trend,
                    )
                    _stat = min(_stat, _pp.pvalue)
                if _stat > 0.05:
                    votes += 1
            except InfeasibleTestException:
                voters -= 1

            try:
                _stat = 1
                for trend in ("c", "ct"):
                    _dfgls = uroot.DFGLS(
                        y,
                        trend=trend,
                        method="aic"
                        if self.lag_criteria not in ("aic", "bic")
                        else self.lag_criteria,
                    )
                    _stat = min(_stat, _dfgls.pvalue)
                if _stat > 0.05:
                    votes += 1
            except InfeasibleTestException:
                voters -= 1

            try:
                _stat = 1
                for trend in ("c", "t", "ct"):
                    _za = uroot.ZivotAndrews(
                        y,
                        trend=trend,
                        method=self.lag_criteria,  # pyright: ignore[reportArgumentType]
                    )
                if _stat > 0.05:
                    votes += 1
            except InfeasibleTestException:
                voters -= 1

            if votes >= voters // 2:
                d += 1
                y = y.diff().dropna()
                continue
            break

        return d, D

    def _seasonal_strength(self, x):
        seasons = 4 if self.freq == "Q" else 12
        stl = STL(x, period=seasons).fit()
        tmp = np.var(stl.resid) / np.var(stl.resid + stl.seasonal)
        return max(0, min(1, 1 - tmp))

    def _is_seasonal(self) -> Literal[0, 4, 12]:
        if self.freq not in ("M", "ME", "Q", "QE"):
            return 0
        seasons = 4 if self.freq in ("Q", "QE") else 12

        d, D = self.d, self.D
        y = self.y.obs.copy()

        if D:
            y = (y - y.shift(seasons)).dropna()

        for _ in range(d):
            y = (y - y.shift(1)).dropna()

        _acf = acf(y, nlags=seasons)
        _sd = 2 / sqrt(len(y))

        if (abs(float(_acf[seasons])) > _sd) and float(
            abs(_acf[seasons]) - abs(_acf[seasons - 1])
        ) >= 0.1:
            return seasons

        return 0

    def fit(
        self,
        criteria: str | None = "aic",
    ):
        if criteria is None:
            criteria = self.criteria
        assert criteria in ("aic", "bic", "hqic"), (
            f"Неверный IC в методе fit: {criteria}!"
        )

        self.d, self.D = self._get_dd()
        if self.D:
            self.seasons = 4 if self.freq == "Q" else 12
        else:
            self.seasons = self._is_seasonal()

        if self.seasons:
            self.max_order = min(self.max_order, self.seasons - 1)

        d, D, s = self.d, self.D, self.seasons

        _init = [(2, d, 2), (0, d, 0), (1, d, 0), (0, d, 1)]
        if s:
            _inits = [(1, D, 1, s), (0, D, 0, s), (1, D, 0, s), (0, D, 1, s)]
        else:
            _inits = [(0, D, 0, 0)]

        _init = list(product(_init, _inits, [False]))

        _endog = self.y.obs

        _aic = float(inf)
        _mod = None

        for spec in _init:
            if spec[0][0] > self.max_order or spec[0][2] > self.max_order:
                continue
            try:
                _crit = (
                    arima.ARIMA(
                        _endog,
                        order=spec[0],
                        seasonal_order=spec[1],
                        trend="n",
                    )
                    .fit()
                    .info_criteria(criteria)
                )
            except LinAlgError:
                _crit = float(inf)

            if _crit < _aic:
                _aic, _mod = _crit, spec

        assert _mod is not None, "Не получилось инициализировать очередь моделей!"
        match d + D:
            case 0:
                _trend = "c"
            case 1:
                _trend = "t"
            case _:
                _trend = "n"

        queue = [(_aic, _mod)]
        seen = set()

        best_aic, best_mod = float("inf"), None

        while queue:
            cur_aic, cur_mod = hq.heappop(queue)
            seen.add(cur_mod)

            if cur_aic > best_aic:
                continue

            best_aic, best_mod = cur_aic, cur_mod
            best_trend = _trend if cur_mod[2] else "n"

            for _nx_mod in self._next_step(cur_mod):
                if _nx_mod in seen:
                    continue
                try:
                    if (
                        _crit := arima.ARIMA(
                            _endog,
                            order=_nx_mod[0],
                            seasonal_order=_nx_mod[1],
                            trend=_trend if _nx_mod[2] else "n",
                        )
                        .fit()
                        .info_criteria(criteria)
                    ) < best_aic:
                        hq.heappush(queue, (_crit, _nx_mod))
                        best_aic = _crit
                        best_trend = _trend if _nx_mod[2] else "n"
                except LinAlgError:
                    continue

        assert best_mod is not None
        best_mod = (best_mod[0], best_mod[1], best_trend)

        return best_mod, arima.ARIMA(
            _endog,
            order=best_mod[0],
            seasonal_order=best_mod[1],
            trend=_trend,
        )
