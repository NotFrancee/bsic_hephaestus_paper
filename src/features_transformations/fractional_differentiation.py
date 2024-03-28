import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def getWeights_FFD(d, thres):
    """
    Get the weights for Fractional Differentiation,
    given the threshold for the weight.
    This is the FFD implementation of Fractional Differentiation.

    Parameters
    ----------
    d : float
        The order of differentiation.
    thres : float
        Threshold that determines the cutoff weight for the window.

    Returns
    -------
    np.ndarray
        The array of weights for the fractional differentiation.
    """

    w, k = [1.0], 1

    while True:
        w_ = -w[-1] / k * (d - k + 1)

        if abs(w_) < thres:
            break

        w.append(w_)
        k += 1

    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def fracDiff_FFD(series, d, thres=1e-5):
    """
    Fractionally Differentiate a Series with the FFD method.

    Parameters
    ----------
    series : pd.Series
        Series to differentiate
    d : float
        The order of differentiation
    thres : float, optional
        The cutoff weight for the window, by default 1e-5
    """

    # 1) Compute weights for the longest series
    w = getWeights_FFD(d, thres)

    # 2) Apply weights to values
    width = len(w)
    df = {}

    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method="ffill").dropna(), pd.Series()

        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue  # exclude NAs
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0]

        df[name] = df_.copy(deep=True)

    df = pd.concat(df, axis=1)

    return df


# def plotMinFFD():
#     path, instName = "./", "ES1_Index_Method12"

#     out = pd.DataFrame(columns=["adfStat", "pVal", "lags", "nObs", "95% conf", "corr"])
#     df0 = pd.read_csv(path + instName + ".csv", index_col=0, parse_dates=True)

#     for d in np.linspace(0, 1, 11):
#         df1 = np.log(df0[["Close"]]).resample("1D").last()  # downcast to daily obs
#         df2 = fracDiff_FFD(df1, d, thres=0.01)
#         corr = np.corrcoef(df1.loc[df2.index, "Close"], df2["Close"])[0, 1]
#         df2 = adfuller(df2["Close"], maxlag=1, regression="c", autolag=None)
#         out.loc[d] = list(df2[:4]) + [df2[4]["5%"]] + [corr]  # with critical value

#     out.to_csv(path + instName + "_testMinFFD.csv")
#     out[["adfStat", "corr"]].plot(secondary_y="adfStat")
#     plt.axhline(out["95% conf"].mean(), linewidth=1, color="r", linestyle="dotted")
#     plt.savefig(path + instName + "_testMinFFD.png")
#     plt.show()
